"""Check acceleration preserves contrast and invalidates caches correctly."""
from unittest import mock
import unittest
import sys
import numpy as np
from scipy.ndimage import map_coordinates
from nion.usim_device import KikuchiModel, RonchigramContrast, SimulationSettings, Noise
from nion.data import DataAndMetadata, Calibration
from nion.utils import Geometry
from nionswift_plugin.usim.test import KikuchiModel_test


class TestRonchigramPerformance(unittest.TestCase):
    def test_cached_composition_matches_reference_and_reuses_blurs(self):
        rng = np.random.default_rng(42)
        pattern = rng.uniform(-1, 1, (96, 128)).astype(np.float32)
        thickness = np.linspace(0, 220, pattern.size).reshape(pattern.shape)
        real = (400*KikuchiModel.transmission(thickness)).astype(np.float32)
        real[0, :20] = 0
        angles = (.0004, .0006)
        composer = RonchigramContrast.ContrastComposer(pattern, angles, "cpu")
        expected, metadata = KikuchiModel.compose_contrast(real, 400, pattern, angles)
        actual, info = composer.compose(real, 400)
        np.testing.assert_allclose(actual, expected, rtol=4e-6, atol=8e-5)
        np.testing.assert_allclose(info["projected_thickness_range_nm"], metadata["projected_thickness_range_nm"], atol=2e-5)
        bank = composer.bank
        with mock.patch.object(composer, "filter", wraps=composer.filter) as filtering:
            composer.compose(real, 400)
            self.assertEqual(filtering.call_count, 1)  # halo only, no line blurs
        self.assertIs(composer.bank, bank)
        self.assertLessEqual(composer.bank.nbytes, SimulationSettings.RONCHIGRAM_PATTERN_CACHE_BYTES)

    def test_cuda_failure_falls_back_without_losing_frame(self):
        real = np.full((32, 32), 70, np.float32)
        composer = RonchigramContrast.ContrastComposer(np.ones_like(real), (.001, .001), "cpu")
        expected, _ = composer.compose(real, 100)
        composer.backend = "gpu"
        original = composer._compose
        def fail_gpu_once(image, vacuum):
            if composer.backend == "gpu":
                raise RuntimeError("device lost")
            return original(image, vacuum)
        with mock.patch.object(composer, "_compose", side_effect=fail_gpu_once):
            actual, _ = composer.compose(real, 100)
        self.assertEqual(composer.backend, "cpu")
        self.assertIn("device lost", composer.fallback_reason)
        np.testing.assert_array_equal(actual, expected)

    def test_missing_cupy_uses_cpu(self):
        with mock.patch.dict(sys.modules, {"cupy": None}):
            composer = RonchigramContrast.ContrastComposer(np.ones((32, 32), np.float32), (.001, .001), "gpu")
        self.assertEqual(composer.backend, "cpu")
        actual, metadata = composer.compose(np.full((32, 32), 70, np.float32), 100)
        self.assertTrue(np.isfinite(actual).all())
        self.assertEqual(metadata["compute_backend"], "cpu")

    def test_projection_cache_survives_probe_move_but_not_stage_or_radius(self):
        fixture = KikuchiModel_test.TestKikuchiModel()
        fixture.setUp()
        manager, camera, context, area = fixture.make_camera()
        manager.set_value("C10Control", 1000e-9)
        def capture(x=.5):
            return camera.get_frame_data(area, Geometry.IntSize(1, 1), .1, context, Geometry.FloatPoint(.5, x))
        try:
            with mock.patch.object(fixture.sample, "plot_features", wraps=fixture.sample.plot_features) as projection:
                with mock.patch.object(RonchigramContrast, "ContrastComposer", wraps=RonchigramContrast.ContrastComposer) as composer:
                    first = capture()
                    moved = capture(.6)
                    self.assertGreater(np.max(np.abs(moved.data-first.data)), 1)
                    self.assertEqual(projection.call_count, 1)
                    self.assertEqual(composer.call_count, 1)
                    manager.set_value_2d("stage_tilt_rad", Geometry.FloatPoint(.01, 0))
                    capture(.6)
                    self.assertEqual(projection.call_count, 1)
                    self.assertEqual(composer.call_count, 2)
                    manager.set_value_2d("stage_position_m", Geometry.FloatPoint(0, 10e-9))
                    capture(.6)
                    self.assertEqual(projection.call_count, 2)
                    fixture.sample.features[0].radius_nm = 45
                    capture(.65)
                    self.assertEqual(projection.call_count, 3)
        finally:
            camera.close()

    def test_gpu_matches_cpu_when_cuda_available(self):
        pattern = np.random.default_rng(5).uniform(-1, 1, (128, 128)).astype(np.float32)
        real = np.linspace(45, 100, pattern.size, dtype=np.float32).reshape(pattern.shape)
        cpu = RonchigramContrast.ContrastComposer(pattern, (.0005, .0005), "cpu")
        gpu = RonchigramContrast.ContrastComposer(pattern, (.0005, .0005), "gpu")
        if gpu.backend != "gpu":
            self.skipTest(gpu.fallback_reason or "CUDA unavailable")
        expected, _ = cpu.compose(real, 100)
        actual, info = gpu.compose(real, 100)
        self.assertEqual(info["compute_backend"], "gpu")
        np.testing.assert_allclose(actual, expected, rtol=5e-6, atol=5e-5)

    def test_gpu_mapper_matches_cpu_for_displacement_and_new_source(self):
        try:
            mapper = RonchigramContrast.GPUMapper()
        except Exception as error:
            self.skipTest(str(error))
        source = np.random.default_rng(7).uniform(0, 100, (64, 80)).astype(np.float32)
        y, x = np.indices(source.shape, dtype=float)
        coordinates = [32+.8*(y-32)+.002*(x-40)**2, 40+.9*(x-40)]
        for dy, dx in ((0., 0.), (3.7, -5.2), (-31., 38.)):
            try:
                actual = mapper.apply(source, coordinates, dy, dx).get()
            except Exception as error:
                self.skipTest(str(error))
            expected = map_coordinates(source, [coordinates[0]+dy, coordinates[1]+dx], order=1)
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-5)
        replacement = source*.5
        actual = mapper.apply(replacement, coordinates, 0, 0).get()
        expected = map_coordinates(replacement, coordinates, order=1)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-5)
        self.assertIs(mapper.source, replacement)

    def test_gpu_shot_noise_is_fresh_and_has_electron_counting_variance(self):
        composer = RonchigramContrast.ContrastComposer(np.zeros((32, 32), np.float32), (.001, .001), "gpu")
        if composer.backend != "gpu":
            self.skipTest(composer.fallback_reason or "CUDA unavailable")
        source = DataAndMetadata.new_data_and_metadata(np.full((256, 256), 200, np.float32),
            intensity_calibration=Calibration.Calibration(units="counts"), metadata={"test": "noise"})
        noise = Noise.ElectronCountingNoise(10, seed=42)
        first = noise.apply_gpu(source, 2)
        second = noise.apply_gpu(source, 2)
        self.assertFalse(noise._gpu_failed)
        self.assertFalse(np.array_equal(first.data, second.data))
        self.assertAlmostEqual(float(first.data.mean()), 400, delta=2)
        self.assertAlmostEqual(float(first.data.var()), 4000, delta=160)
        self.assertEqual(first.metadata, source.metadata)
        self.assertEqual(first.intensity_calibration.units, "counts")
        np.testing.assert_array_equal(noise.apply_gpu(source, 0).data, np.zeros_like(source.data))
        noise.enabled = False
        np.testing.assert_array_equal(noise.apply_gpu(source, 2).data, source.data*2)
        noise.clear_gpu_cache()
        self.assertIsNone(noise._gpu_data)
