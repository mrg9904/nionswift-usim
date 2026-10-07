"""Numerical regressions for the optimized STL and depth-filter paths."""

import unittest
from unittest import mock

import numpy
import trimesh

from nion.usim_device import HAADFFocusModel
from nion.usim_device import SimulationSettings
from nion.usim_device import STLDepthSample
from nion.usim_device import SurfaceRasterizer
from nion.utils import Geometry


class TestHAADFPerformance(unittest.TestCase):
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
        for fast_boxes in (False, True):
            for defocus in (0.0, 25e-9, 500e-9):
                with self.subTest(fast_boxes=fast_boxes, defocus=defocus), mock.patch.object(
                    SimulationSettings, "USE_FAST_BOX_FILTER", fast_boxes
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


if __name__ == "__main__":
    unittest.main()
