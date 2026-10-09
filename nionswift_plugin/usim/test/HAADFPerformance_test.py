"""Numerical regressions for the optimized STL and depth-filter paths."""

import unittest
from unittest import mock

import numpy
import trimesh
import scipy.ndimage
import scipy.fft

from nion.usim_device import HAADFFocusModel
from nion.usim_device import SimulationSettings
from nion.usim_device import STLDepthSample
from nion.usim_device import SurfaceRasterizer
from nion.usim_device import Noise
from nion.utils import Geometry


class TestHAADFPerformance(unittest.TestCase):
    def test_large_defocus_keeps_changing_without_integer_filter_steps(self) -> None:
        image = numpy.zeros((96, 128), dtype=numpy.float32)
        image[25:65, 32:74] = 1.0
        args = dict(best_focus_m=0.0, convergence_angle_rad=0.04,
                    pixel_size_y_nm=0.5, pixel_size_x_nm=0.5)
        for sign in (-1, 1):
            outputs = [HAADFFocusModel.apply_defocus(image, defocus_m=sign * nm * 1e-9, **args)
                       for nm in (200.0, 500.0, 501.0, 1000.0)]
            for before, after in zip(outputs, outputs[1:]):
                self.assertGreater(float(numpy.max(numpy.abs(after - before))), 1e-6)
                self.assertLess(float(after.std()), float(before.std()))
                self.assertAlmostEqual(float(after.mean()), float(image.mean()), places=6)

    def test_spectral_gaussian_matches_reflect_spatial_reference(self) -> None:
        image = numpy.zeros((47, 63), dtype=numpy.float32)
        image[:5, :7] = 1.0  # corner feature detects unwanted periodic wrapping
        for sigma in ((3.0, 3.0), (5.0, 12.0), (25.0, 40.0)):
            actual = HAADFFocusModel.fast_gaussian_filter(image, *sigma)
            expected = scipy.ndimage.gaussian_filter(image, sigma=sigma, mode="reflect", truncate=5.0)
            numpy.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)

    def test_depth_spectra_are_reused_and_bounded(self) -> None:
        planes = HAADFFocusModel.PreparedDepthPlanes([
            (0.0, numpy.eye(32, dtype=numpy.float32)),
            (10.0, numpy.fliplr(numpy.eye(32, dtype=numpy.float32)).copy()),
        ])
        args = dict(best_focus_m=0.0, convergence_angle_rad=0.04,
                    pixel_size_y_nm=0.5, pixel_size_x_nm=0.5)
        with mock.patch.object(scipy.fft, "dctn", wraps=scipy.fft.dctn) as transform:
            for nm in (500, 510, 1000):
                HAADFFocusModel.apply_depth_planes_defocus(planes, defocus_m=nm * 1e-9, **args)
            self.assertEqual(transform.call_count, len(planes))
        self.assertFalse(planes[0][1].flags.writeable)
        with mock.patch.object(SimulationSettings, "DEPTH_SPECTRUM_CACHE_BYTES", 0):
            uncached = HAADFFocusModel.PreparedDepthPlanes([(0.0, numpy.eye(16, dtype=numpy.float32))])
            uncached.spectrum(0)
            self.assertEqual(uncached._cached_bytes, 0)

    def test_tilted_surfaces_match_vertical_rays(self) -> None:
        mesh = trimesh.creation.box(extents=(10, 12, 8))
        mesh.apply_transform(trimesh.transformations.rotation_matrix(0.31, (1, 1, 0)))
        x = numpy.linspace(-8, 8, 53)
        y = numpy.linspace(-9, 9, 47)
        with mock.patch.object(SimulationSettings, "STL_SURFACE_BACKEND", "cpu"):
            lower, upper = SurfaceRasterizer.SurfaceRasterizer(mesh.triangles).surface_maps(x, y)
        xx, yy = numpy.meshgrid(x, y)
        origins = numpy.column_stack((xx.ravel(), yy.ravel(), numpy.full(xx.size, 20.0)))
        directions = numpy.zeros_like(origins)
        directions[:, 2] = -1
        hits, indices, _ = mesh.ray.intersects_location(origins, directions, multiple_hits=True)
        expected_lower = numpy.full(xx.size, numpy.inf)
        expected_upper = numpy.full(xx.size, -numpy.inf)
        numpy.minimum.at(expected_lower, indices, hits[:, 2])
        numpy.maximum.at(expected_upper, indices, hits[:, 2])
        for actual, expected in ((lower, expected_lower), (upper, expected_upper)):
            expected[~numpy.isfinite(expected)] = numpy.nan
            numpy.testing.assert_allclose(actual, expected.reshape(xx.shape), atol=1e-6, rtol=1e-6)

    def test_real_stl_planes_match_after_fov_and_stage_changes(self) -> None:
        with mock.patch.object(SimulationSettings, "STL_SURFACE_BACKEND", "cpu"):
            fast = STLDepthSample.STLDepthSample(1000)
            reference = STLDepthSample.STLDepthSample(1000)
        for fov, offset in ((80.0, Geometry.FloatPoint()), (200.0, Geometry.FloatPoint()),
                            (4000.0, Geometry.FloatPoint(30e-9, -70e-9))):
            with self.subTest(fov=fov):
                args = (offset, Geometry.FloatSize(fov, fov * 1.2), Geometry.FloatPoint(),
                        Geometry.FloatPoint(3, -2), Geometry.IntSize(48, 64))
                thickness = SimulationSettings.calculate_depth_slice_thickness_nm(fov, fov * 1.2)
                with mock.patch.object(SimulationSettings, "STL_USE_SURFACE_RASTERIZER", False):
                    expected = reference.generate_depth_planes(*args, thickness)
                with mock.patch.object(SimulationSettings, "STL_USE_SURFACE_RASTERIZER", True):
                    actual = fast.generate_depth_planes(*args, thickness)
                self.assertEqual(len(actual), len(expected))
                for (depth, plane), (expected_depth, expected_plane) in zip(actual, expected):
                    self.assertEqual(depth, expected_depth)
                    numpy.testing.assert_allclose(plane, expected_plane, atol=1e-6, rtol=1e-5)

    def test_grouped_filters_match_separate_filters_and_preserve_inputs(self) -> None:
        rng = numpy.random.default_rng(11)
        planes = [(depth, rng.random((32, 48), dtype=numpy.float32))
                  for depth in (0.0, 1.0, 10.0, 30.0, 50.0, 100.0)]
        original = [plane.copy() for _, plane in planes]
        for threshold in (float("inf"), 3.0):
            for defocus in (0.0, 25e-9, 500e-9):
                with self.subTest(threshold=threshold, defocus=defocus), mock.patch.object(
                    SimulationSettings, "FOURIER_BLUR_THRESHOLD_PX", threshold
                ):
                    args = dict(defocus_m=defocus, best_focus_m=0.0, convergence_angle_rad=0.04,
                                pixel_size_y_nm=0.5, pixel_size_x_nm=0.75)
                    actual = HAADFFocusModel.apply_depth_planes_defocus(planes, **args)
                    expected = numpy.zeros((32, 48), dtype=numpy.float32)
                    for depth, plane in planes:
                        expected += HAADFFocusModel.apply_defocus(
                            plane, **dict(args, best_focus_m=depth * 1e-9)
                        )
                    numpy.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
        for (_, plane), before in zip(planes, original):
            numpy.testing.assert_array_equal(plane, before)

    def test_empty_fov_returns_empty_plane(self) -> None:
        with mock.patch.object(SimulationSettings, "STL_SURFACE_BACKEND", "cpu"):
            sample = STLDepthSample.STLDepthSample(1000)
        planes = sample.generate_depth_planes(
            Geometry.FloatPoint(), Geometry.FloatSize(100, 100), Geometry.FloatPoint(),
            Geometry.FloatPoint(100000, 100000), Geometry.IntSize(16, 16), 1.0,
        )
        self.assertEqual(len(planes), 1)
        self.assertFalse(numpy.any(planes[0][1]))


class TestHAADFNoise(unittest.TestCase):
    def test_dose_mean_and_relative_noise(self) -> None:
        noise = Noise.HAADFNoise(seed=21)
        image = numpy.ones((512, 512), dtype=numpy.float32)
        rate = 200e-12 / 1.602176634e-19 * 1e-6 * SimulationSettings.HAADF_DETECTION_EFFICIENCY
        for dwell in (1., 4., 16.):
            normalized = noise.apply(image, dwell, 200e-12) / dwell
            self.assertAlmostEqual(float(normalized.mean()), 1., places=3)
            expected = 1 / (rate * dwell) + (SimulationSettings.HAADF_READ_NOISE_ELECTRONS / (rate * dwell)) ** 2
            self.assertAlmostEqual(float(normalized.var()) / expected, 1., delta=.025)
        doubled = noise.apply(image, 1., 400e-12)
        self.assertAlmostEqual(float(doubled.mean()), 2., places=3)

    def test_noise_can_be_disabled_and_inputs_are_preserved(self) -> None:
        image = numpy.eye(8, dtype=numpy.float32)
        with mock.patch.object(SimulationSettings, "HAADF_SHOT_NOISE_ENABLED", False), mock.patch.object(
            SimulationSettings, "HAADF_READ_NOISE_ELECTRONS", 0.
        ):
            numpy.testing.assert_array_equal(Noise.HAADFNoise().apply(image, 4., 200e-12), image * 4)
        numpy.testing.assert_array_equal(image, numpy.eye(8, dtype=numpy.float32))


class TestGPUHAADF(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import cupy
            if not cupy.cuda.runtime.getDeviceCount():
                raise unittest.SkipTest("No CUDA GPU")
        except (ImportError, RuntimeError) as error:
            raise unittest.SkipTest(str(error))

    def test_gpu_matches_cpu_after_geometry_and_focus_changes(self) -> None:
        with mock.patch.object(SimulationSettings, "STL_SURFACE_BACKEND", "cpu"):
            cpu = STLDepthSample.STLDepthSample(1000)
        with mock.patch.object(SimulationSettings, "STL_SURFACE_BACKEND", "gpu"):
            gpu = STLDepthSample.STLDepthSample(1000)
            for normalized in (True, False):
                with mock.patch.object(SimulationSettings, "STL_NORMALIZE_COLUMN_INTENSITY", normalized):
                    for fov, center in ((80., Geometry.FloatPoint()), (200., Geometry.FloatPoint(3, -2)),
                                        (100., Geometry.FloatPoint(100000, 100000))):
                        args = (Geometry.FloatPoint(), Geometry.FloatSize(fov, fov * 1.2),
                                Geometry.FloatPoint(), center, Geometry.IntSize(47, 63), fov / 100)
                        expected_planes = cpu.generate_depth_planes(*args)
                        actual_planes = gpu.generate_depth_planes(*args)
                        self.assertTrue(getattr(actual_planes, "_on_gpu", False))
                        for focus in (-500., 0., 25., 500., 501.):
                            kwargs = dict(defocus_m=focus * 1e-9, best_focus_m=0., convergence_angle_rad=.04,
                                          pixel_size_y_nm=fov / 47, pixel_size_x_nm=fov * 1.2 / 63)
                            expected = HAADFFocusModel.apply_depth_planes_defocus(expected_planes, **kwargs)
                            actual = HAADFFocusModel.apply_depth_planes_defocus(actual_planes, **kwargs)
                            numpy.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-5)

    def test_gpu_noise_is_device_resident_and_obeys_dose_statistics(self) -> None:
        import cupy
        noise = Noise.HAADFNoise(seed=21)
        image = cupy.ones((512, 512), dtype=cupy.float32)
        rate = 200e-12 / 1.602176634e-19 * 1e-6 * SimulationSettings.HAADF_DETECTION_EFFICIENCY
        for dwell in (1., 4., 16.):
            output = noise.apply(image, dwell, 200e-12)
            self.assertIsInstance(output, cupy.ndarray)
            normalized = output / dwell
            self.assertAlmostEqual(float(normalized.mean()), 1., places=3)
            expected = 1 / (rate * dwell) + (SimulationSettings.HAADF_READ_NOISE_ELECTRONS / (rate * dwell)) ** 2
            self.assertAlmostEqual(float(normalized.var()) / expected, 1., delta=.025)

    def test_scan_generator_returns_numpy_for_gpu_and_rotated_scans(self) -> None:
        from nion.usim_device import InstrumentDevice
        from nion.device_kit import ScanDevice
        import types
        instrument = types.SimpleNamespace(
            value_manager=types.SimpleNamespace(actual_offset_m=Geometry.FloatPoint()),
            GetVal2D=lambda key: Geometry.FloatPoint(),
            GetVal=lambda key: {"C10Control": 500e-9, "stage_z_m": 0., "ConvergenceAngle": .04,
                                "BeamCurrent": 200e-12}[key])
        with mock.patch.object(SimulationSettings, "STL_SURFACE_BACKEND", "gpu"):
            generator = InstrumentDevice.ScanDataGenerator(sample_index=6)
            for rotation in (0., .3):
                parameters = ScanDevice.ScanFrameParameters(
                    size=(47, 63), pixel_time_us=4., fov_nm=100., rotation_rad=rotation)
                result = generator.generate_scan_data(instrument, parameters)
                self.assertIsInstance(result, numpy.ndarray)
                self.assertEqual(result.shape, (47, 63))
                self.assertEqual(result.dtype, numpy.float32)
                self.assertTrue(numpy.all(numpy.isfinite(result)))


if __name__ == "__main__":
    unittest.main()
