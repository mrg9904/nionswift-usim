"""Physical and numerical checks for geometry-driven EELS."""
import math
import unittest
from types import SimpleNamespace
from unittest import mock
import numpy as np
import trimesh
from nion.data import DataAndMetadata
from nion.usim_device import Noise, EELSCameraSimulator, InstrumentDevice
from nion.instrumentation import stem_controller
from nion.utils import Event
from nion.usim_device import EELSModel, SampleSimulator, STLDepthSample, SurfaceRasterizer, SimulationSettings
from nion.utils import Geometry


class TestEELSModel(unittest.TestCase):
    def test_sphere_thickness_from_center_to_edge_and_bounding_box_vacuum(self):
        sample = SampleSimulator.SphericalParticleSample(1000)
        for radius in (0, 20, 40, 49, 50, 55):
            layers = sample.eels_layers_at(Geometry.FloatPoint(x=radius*1e-9, y=0))
            actual = sum(layer.thickness_nm for layer in layers)
            self.assertAlmostEqual(actual, 2*math.sqrt(max(0, 50**2-radius**2)))
        # Within the bounding rectangle but outside the actual sphere.
        self.assertEqual(sample.eels_layers_at(Geometry.FloatPoint(45e-9, 45e-9)), [])

    def test_thickness_blocks_use_20_50_100_nm(self):
        sample = SampleSimulator.ThreeThicknessBlocksSample(1000)
        for x, thickness in ((-60, 20), (0, 50), (60, 100)):
            self.assertEqual(sample.eels_layers_at(Geometry.FloatPoint(x=x*1e-9, y=0))[0].thickness_nm, thickness)

    def test_stl_uses_surface_thickness_and_vacuum(self):
        mesh = trimesh.creation.box(extents=(40, 40, 25))
        mesh.apply_translation((100, -50, 40))
        sample = STLDepthSample.STLDepthSample.__new__(STLDepthSample.STLDepthSample)
        with mock.patch.object(SimulationSettings, 'STL_SURFACE_BACKEND', 'cpu'):
            sample._STLDepthSample__rasterizer = SurfaceRasterizer.SurfaceRasterizer(mesh.triangles)
        self.assertAlmostEqual(sample.eels_layers_at(Geometry.FloatPoint(x=100e-9, y=-50e-9))[0].thickness_nm, 25)
        self.assertEqual(sample.eels_layers_at(Geometry.FloatPoint()), [])

    def test_probe_coordinates_include_stage_shift_fov_center_and_rotation(self):
        args = (Geometry.FloatPoint(y=2e-9, x=3e-9), Geometry.FloatSize(100, 200),
                Geometry.FloatPoint(y=20, x=10), Geometry.FloatPoint(y=.5, x=.75))
        point = EELSModel.probe_sample_position(*args, rotation_rad=math.pi/2)
        self.assertAlmostEqual(point.x, 7e-9)
        self.assertAlmostEqual(point.y, 68e-9)

    def test_plural_scattering_peak_areas_follow_poisson_law(self):
        # Narrow, well-separated plasmons expose scattering orders quantitatively.
        material = EELSModel.EELSMaterial(edges=(), plasmon_eV=100, core_fraction=0)
        energies = np.arange(-20, 2000, .25)
        for tau in (.2, 1., 2.):
            data, info = EELSModel.spectrum_probabilities([EELSModel.EELSLayer(tau*100, material)], energies, .25)
            self.assertAlmostEqual(data.sum(), 1, places=7)
            zlp = data[np.abs(energies) < 4].sum()
            first = data[(energies > 55) & (energies < 145)].sum()
            second = data[(energies > 150) & (energies < 250)].sum()
            self.assertAlmostEqual(zlp, math.exp(-tau), places=6)
            self.assertAlmostEqual(first, tau*math.exp(-tau), delta=2e-4)
            self.assertAlmostEqual(second/first, tau/2, delta=.002)
            self.assertAlmostEqual(info['t_over_lambda'], tau)
            self.assertLess(info['poisson_tail_probability'], 1e-10)

    def test_spherical_low_loss_thickness_and_zlp_change_monotonically(self):
        sample = SampleSimulator.SphericalParticleSample(1000)
        energies = np.arange(-20, 1024, .5)
        previous_zlp = 0
        previous_tau = float('inf')
        for radius in (0, 20, 40, 49, 50):
            layers = sample.eels_layers_at(Geometry.FloatPoint(x=radius*1e-9, y=0))
            data, info = EELSModel.spectrum_probabilities(layers, energies, .5)
            self.assertLessEqual(info['t_over_lambda'], previous_tau)
            self.assertGreater(info['zero_loss_fraction'], previous_zlp)
            self.assertAlmostEqual(float(data[np.abs(energies) < 4].sum()), info['zero_loss_fraction'], delta=.004)
            previous_zlp = info['zero_loss_fraction']
            previous_tau = info['t_over_lambda']

    def test_energy_windows_and_dispersion_preserve_absolute_counts(self):
        layers = [EELSModel.EELSLayer(100, EELSModel.EELSMaterial())]
        full, _ = EELSModel.spectrum_probabilities(layers, np.arange(-20, 1024, .5), .5)
        cropped, _ = EELSModel.spectrum_probabilities(layers, np.arange(100, 1024, .5), .5)
        np.testing.assert_allclose(cropped, full[240:], atol=1e-14)
        self.assertLess(cropped.sum(), .1)
        # Coarse channels integrate rather than point-sample the ZLP.
        vacuum, _ = EELSModel.spectrum_probabilities([], np.arange(-20, 100, 10), 10)
        self.assertAlmostEqual(vacuum.sum(), 1)
        self.assertAlmostEqual(vacuum[2], 1)

    def test_layer_additivity_and_nonunit_mean_free_path(self):
        material = EELSModel.EELSMaterial(mean_free_path_nm=50)
        energies = np.arange(-20, 1000, .5)
        one, _ = EELSModel.spectrum_probabilities([EELSModel.EELSLayer(60, material)], energies, .5)
        two, info = EELSModel.spectrum_probabilities([EELSModel.EELSLayer(20, material), EELSModel.EELSLayer(40, material)], energies, .5)
        np.testing.assert_allclose(one, two, atol=1e-14)
        self.assertEqual(info['thickness_nm'], 60)
        self.assertAlmostEqual(info['t_over_lambda'], 1.2)

    def test_invalid_physical_parameters_are_rejected(self):
        for layer in (EELSModel.EELSLayer(-1, EELSModel.EELSMaterial()),
                      EELSModel.EELSLayer(1, EELSModel.EELSMaterial(mean_free_path_nm=0)),
                      EELSModel.EELSLayer(float('nan'), EELSModel.EELSMaterial())):
            with self.assertRaises(ValueError):
                EELSModel.spectrum_probabilities([layer], np.arange(10), 1)

    def test_eels_shot_noise_keeps_geometry_metadata_and_correct_variance(self):
        gain = 10
        source = DataAndMetadata.new_data_and_metadata(np.full(100000, 100., dtype=np.float32),
            metadata={'eels_simulation': {'thickness_nm': 100}})
        noisy = Noise.EELSShotNoise(gain, seed=1).apply(source)
        self.assertEqual(noisy.metadata, source.metadata)
        self.assertGreaterEqual(noisy.data.min(), 0)
        self.assertAlmostEqual(noisy.data.mean(), 100., delta=.4)
        self.assertAlmostEqual(noisy.data.var(), 100*gain, delta=20)
        zero = DataAndMetadata.new_data_and_metadata(np.zeros(100, dtype=np.float32))
        self.assertEqual(Noise.EELSShotNoise(gain).apply(zero).data.sum(), 0)

    def test_camera_geometry_metadata_dose_zero_beam_and_energy_binning(self):
        manager = InstrumentDevice.ValueManager()
        manager.set_value_2d('stage_position_m', Geometry.FloatPoint())
        beam = [200e-12]
        device = SimpleNamespace(value_manager=manager, GetVal=lambda _: beam[0],
            scan_controller=None, probe_state_changed_event=Event.Event(),
            scan_data_generator=SimpleNamespace(sample=SampleSimulator.SphericalParticleSample(1000)))
        camera = EELSCameraSimulator.EELSCameraSimulator(device, Geometry.IntSize(8, 1024), 10)
        camera.noise.enabled = False
        context = stem_controller.ScanContext(Geometry.IntSize(64, 64), Geometry.FloatPoint(), 160., 0.)
        area = Geometry.IntRect(origin=Geometry.IntPoint(), size=Geometry.IntSize(8, 1024))
        def capture(exposure=.01, bins=Geometry.IntSize(1, 1)):
            return camera.get_frame_data(area, bins, exposure, context, None)
        try:
            center = capture()
            self.assertAlmostEqual(center.metadata['eels_simulation']['thickness_nm'], 100)
            np.testing.assert_allclose(capture(.02).data, center.data*2, rtol=2e-7)
            binned = capture(bins=Geometry.IntSize(1, 4))
            np.testing.assert_allclose(binned.data, center.data.reshape(8, 256, 4).sum(axis=2), rtol=2e-7)
            self.assertEqual(binned.dimensional_calibrations[-1].scale, 2)
            self.assertEqual(binned.dimensional_calibrations[-1].offset, -19.25)
            edge_zlp = []
            for radius in (0, 20, 40, 49, 55):
                manager.set_value_2d('stage_position_m', Geometry.FloatPoint(x=-radius*1e-9, y=0))
                frame = capture()
                info = frame.metadata['eels_simulation']
                self.assertAlmostEqual(info['thickness_nm'], 2*math.sqrt(max(0, 50**2-radius**2)))
                edge_zlp.append(info['zero_loss_fraction'])
                self.assertAlmostEqual(float(frame.data.sum()) / camera.get_total_counts(.01), info['detector_window_fraction'], places=6)
            self.assertTrue(np.all(np.diff(edge_zlp) > 0))
            beam[0] = 0
            self.assertEqual(capture(.03).data.sum(), 0)
            beam[0] = 200e-12
            camera.noise.enabled = True
            self.assertIn('eels_simulation', capture(.04).metadata)
        finally:
            camera.close()

    def test_real_stl_eels_matches_haadf_projection_at_pixel_centers(self):
        with mock.patch.object(SimulationSettings, 'STL_SURFACE_BACKEND', 'cpu'):
            sample = STLDepthSample.STLDepthSample(1000)
        image = np.zeros((16, 16), dtype=np.float32)
        fov = 200.
        sample.plot_features(image, Geometry.FloatPoint(), Geometry.FloatSize(fov, fov),
            Geometry.FloatPoint(), Geometry.FloatPoint(), Geometry.IntSize(16, 16))
        occupied = np.argwhere(image > 0)
        self.assertGreater(len(occupied), 0)
        for y, x in occupied[::max(1, len(occupied)//8)]:
            position = Geometry.FloatPoint(x=((x+.5)*fov/16-fov/2)*1e-9,
                                          y=((y+.5)*fov/16-fov/2)*1e-9)
            layers = sample.eels_layers_at(position)
            self.assertAlmostEqual(sum(layer.thickness_nm for layer in layers),
                float(image[y,x])*SimulationSettings.STL_REFERENCE_THICKNESS_NM, places=5)
