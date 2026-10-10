import unittest
from unittest import mock
import numpy as np
from scipy.ndimage import map_coordinates
from nion.usim_device import ParticleProjection, SimulationSettings
from nion.usim_device import RonchigramContrast
from scipy.ndimage import gaussian_filter


class TestParticleProjection(unittest.TestCase):
    def test_batched_halos_preserve_independent_channels_and_boundaries(self):
        try:
            import cupy as cp
            if not cp.cuda.runtime.getDeviceCount():
                self.skipTest('CUDA unavailable')
        except ImportError:
            self.skipTest('CuPy unavailable')
        rng = np.random.default_rng(7)
        images = [np.zeros((24, 31), np.float32), np.full((24, 31), 100, np.float32),
                  rng.uniform(0, 100, (24, 31)).astype(np.float32), rng.uniform(0, 100, (31, 24)).astype(np.float32)]
        angles = (.0008, .0011)
        halos = RonchigramContrast.batched_halos(images, angles, 100)
        for image, halo in zip(images, halos):
            self.assertIsNotNone(halo)
            expected = gaussian_filter(image, SimulationSettings.RONCHIGRAM_DIFFUSE_SIGMA_RAD/np.array(angles), mode='nearest')
            np.testing.assert_allclose(halo, expected, rtol=1e-6, atol=2e-5)

    def test_shared_mapping_matches_independent_channels_with_overlaps_and_binning(self):
        rng = np.random.default_rng(42)
        pieces = [(0, (slice(2, 6), slice(3, 8)), np.zeros((4, 5)), rng.uniform(0, 200, (4, 5))),
                  (1, (slice(3, 8), slice(4, 9)), np.full((5, 5), 300.), rng.uniform(350, 500, (5, 5)))]
        coordinates = [rng.uniform(-1, 5, (7, 8)), rng.uniform(-1, 6, (7, 8))]
        for backend in ('cpu', 'gpu'):
            if backend == 'gpu':
                try:
                    import cupy as cp
                    if not cp.cuda.runtime.getDeviceCount():
                        continue
                except ImportError:
                    continue
            with mock.patch.object(SimulationSettings, 'RONCHIGRAM_BACKEND', backend):
                mapped, inside, actual_backend = ParticleProjection.mapped_particles(pieces, (10, 12), coordinates, (2, 2))
            self.assertEqual(actual_backend, backend)
            for identifier, region, lo, hi in pieces:
                source = np.zeros((10, 12), np.float32)
                source[region] = -100*np.expm1(-(hi-lo)/SimulationSettings.RONCHIGRAM_TRANSMISSION_LENGTH_NM)
                binned = source.reshape(5, 2, 6, 2).sum(axis=(1, 3))
                expected = map_coordinates(binned, coordinates, order=1)
                actual = np.zeros((7, 8), np.float32)
                pixels, values = mapped[identifier]
                actual.ravel()[pixels] = values
                np.testing.assert_allclose(actual, expected, atol=5e-5)
            np.testing.assert_array_equal(inside, (coordinates[0] >= 0) & (coordinates[0] <= 4) &
                                          (coordinates[1] >= 0) & (coordinates[1] <= 5))
