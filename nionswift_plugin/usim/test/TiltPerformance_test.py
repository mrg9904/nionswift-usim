"""Numerical checks for FoV culling, CUDA lines and composite depth batching."""
import unittest
from unittest import mock
import numpy as np
import trimesh
from nion.utils import Geometry
from nion.usim_device import SampleGeometry, SurfaceRasterizer, TiltedSurfaceRasterizer, SimulationSettings
from nion.usim_device import SingleCathodeSample, HAADFFocusModel, KikuchiModel, RonchigramContrast


def require_cuda(test):
    try:
        import cupy as cp
        if not cp.cuda.runtime.getDeviceCount():
            test.skipTest('CUDA unavailable')
    except (ImportError, RuntimeError):
        test.skipTest('CUDA unavailable')


class TestTiltPerformance(unittest.TestCase):
    def test_culled_projection_matches_full_rotated_mesh_including_side_faces_and_edge_on(self):
        meshes = [trimesh.creation.box(extents=(90., 70., 20.)),
            trimesh.creation.box(extents=(150., 100., 30.), transform=trimesh.transformations.translation_matrix([500, 0, 50]))]
        triangles = trimesh.util.concatenate(meshes).triangles
        pivot = np.array([10., 20., 5.])
        raster = TiltedSurfaceRasterizer.TiltedSurfaceRasterizer(triangles, pivot)
        with mock.patch.object(SimulationSettings, 'STL_SURFACE_BACKEND', 'cpu'):
            for tilt in (Geometry.FloatPoint(), Geometry.FloatPoint(x=.4, y=-.3), Geometry.FloatPoint(y=np.pi/2)):
                raster.set_tilt(tilt)
                reference = SurfaceRasterizer.SurfaceRasterizer(SampleGeometry.rotate_triangles(triangles, tilt, pivot))
                for position in (-20., 490., 10000.):
                    x = np.linspace(position-80, position+80, 63)
                    y = np.linspace(-110, 110, 59)
                    expected = reference.surface_maps(x, y)
                    actual = raster.surface_maps(x, y)
                    for a, b in zip(actual, expected):
                        np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-6, equal_nan=True)
            self.assertLessEqual(len(raster.cache), 4)

    def test_cuda_bragg_cones_match_cpu_lines_for_tilted_crystal(self):
        require_cuda(self)
        from pathlib import Path
        crystal = KikuchiModel.load_crystal(str(Path(SingleCathodeSample.__file__).with_name('samples')/'NNMTO_pristine.cif'))
        renderer = RonchigramContrast.GPULineRenderer()
        x, y = np.linspace(-.1, .1, 257), np.linspace(-.06, .08, 193)
        for tx, ty in ((0., 0.), (.2, -.1)):
            bands = KikuchiModel.bands(crystal, 100000, tx_rad=tx, ty_rad=ty, max_angle_rad=.15)
            expected = KikuchiModel.render_lines(bands, x, y)
            actual = renderer.render(bands, x, y)
            np.testing.assert_allclose(actual, expected, atol=8e-6, rtol=8e-6)
        np.testing.assert_array_equal(renderer.render([], x, y), np.zeros((len(y), len(x))))
        from scipy.ndimage import map_coordinates
        yy, xx = np.meshgrid(np.linspace(0, len(y)-1, 512), np.linspace(0, len(x)-1, 640), indexing='ij')
        expected = map_coordinates(KikuchiModel.render_lines(bands, x, y), [yy, xx], order=1)
        np.testing.assert_allclose(renderer.render(bands, x, y, output_shape=(512, 640)), expected, atol=8e-6, rtol=8e-6)

    def test_batched_cuda_depth_matches_cpu_near_and_far_focus_without_filling_gap(self):
        require_cuda(self)
        with mock.patch.object(SimulationSettings, 'STL_SURFACE_BACKEND', 'cpu'):
            sample = SingleCathodeSample.SingleCathodeSample(1000)
            sample.set_stage_tilt(Geometry.FloatPoint(x=.2, y=-.1))
            stage, _ = sample.initial_view
            args = (stage, Geometry.FloatSize(600, 600), Geometry.FloatPoint(), Geometry.FloatPoint(), Geometry.IntSize(64, 64), 2.)
            cpu = sample.generate_depth_planes(*args)
        sample._planes_key = None
        with mock.patch.object(SimulationSettings, 'STL_SURFACE_BACKEND', 'gpu'):
            gpu = sample.generate_depth_planes(*args)
        self.assertTrue(getattr(gpu, '_on_gpu', False))
        cpu_sum = sum(plane for _, plane in cpu)
        gpu_sum = sum(plane for _, plane in gpu)
        np.testing.assert_allclose(gpu_sum, cpu_sum, atol=2e-5, rtol=2e-5)
        for focus in (187e-9, 1000e-9, -500e-9):
            settings = dict(defocus_m=focus, best_focus_m=0., convergence_angle_rad=.04,
                pixel_size_y_nm=600/64, pixel_size_x_nm=600/64)
            expected = HAADFFocusModel.apply_depth_planes_defocus(cpu, **settings)
            actual = HAADFFocusModel.apply_depth_planes_defocus(gpu, **settings)
            np.testing.assert_allclose(actual, expected, atol=4e-5, rtol=4e-5)
        self.assertLessEqual(sum(layer._cached_bytes for layer in gpu.layers), len(gpu.layers)*SimulationSettings.GPU_DEPTH_SPECTRUM_CACHE_BYTES)
