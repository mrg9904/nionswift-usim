"""Composite thickness, crystal orientation, and real acquisition profile tests."""
import unittest
from unittest import mock
import numpy as np
from nion.utils import Geometry
from nion.device_kit import ScanDevice
from nion.instrumentation.test import AcquisitionTestContext
from nion.swift.test import TestContext
from nion.usim_device import SingleCathodeSample, SimulationSettings, KikuchiModel, EELSCameraSimulator, InstrumentDevice, DeviceConfiguration
from nionswift_plugin.usim.test import KikuchiModel_test


class TestSingleCathode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with mock.patch.object(SimulationSettings, 'STL_SURFACE_BACKEND', 'cpu'):
            cls.sample = SingleCathodeSample.SingleCathodeSample(1000)

    def setUp(self):
        self.sample.set_stage_tilt(Geometry.FloatPoint())

    def test_crystal_001_aligns_with_prism_normal_and_stage_tilt_rotates_it(self):
        crystal = KikuchiModel.load_crystal(self.sample.crystal_cif_path)
        axis = crystal.cell_angstrom @ np.array([0., 0., 1.])
        axis /= np.linalg.norm(axis)
        rotation = KikuchiModel.orientation_matrix(crystal, sample_rotation=self.sample.crystal_rotation)
        np.testing.assert_allclose(rotation @ axis, self.sample.model_info['base_normal_unit_xyz'], atol=1e-8)
        tilted = KikuchiModel.orientation_matrix(crystal, tx_rad=.1, sample_rotation=self.sample.crystal_rotation)
        self.assertGreater(np.linalg.norm(tilted @ axis-rotation @ axis), .01)

    def test_material_thickness_excludes_air_gap_and_depth_planes_preserve_it(self):
        position = Geometry.FloatPoint(x=self.sample.center_nm[0]*1e-9, y=self.sample.center_nm[1]*1e-9)
        layers = self.sample.eels_layers_at(position)
        self.assertEqual(len(layers), 2)
        self.assertAlmostEqual(layers[0].thickness_nm, 5.)
        offset, _ = self.sample.initial_view
        maps = self.sample._maps(offset, Geometry.FloatSize(1, 1), Geometry.FloatPoint(), Geometry.FloatPoint(), Geometry.IntSize(1, 1))
        _, _, bottom, top = maps
        self.assertGreater(bottom[0, 0]-5., 10.)
        thickness = sum(l.thickness_nm for l in layers)
        self.assertAlmostEqual(thickness, 5.+top[0, 0]-bottom[0, 0], places=4)
        self.assertLess(thickness, top[0, 0])
        planes = self.sample.generate_depth_planes(offset, Geometry.FloatSize(1, 1), Geometry.FloatPoint(), Geometry.FloatPoint(), Geometry.IntSize(1, 1), 2.)
        self.assertAlmostEqual(sum(float(data[0, 0])*20 for _, data in planes), thickness, places=4)

    def test_live_eels_haadf_and_kikuchi_on_cathode_but_not_on_carbon(self):
        fixture = KikuchiModel_test.TestKikuchiModel()
        fixture.sample = self.sample
        manager, camera, context, area = fixture.make_camera(128)
        eels = EELSCameraSimulator.EELSCameraSimulator(camera.instrument, Geometry.IntSize(8, 2048), 10)
        eels.noise.enabled = False
        ea = Geometry.IntRect(origin=Geometry.IntPoint(), size=Geometry.IntSize(8, 2048))
        try:
            stage, _ = self.sample.initial_view
            manager.set_value_2d('stage_position_m', stage)
            manager.set_value('C10Control', 1000e-9)
            capture = lambda: camera.get_frame_data(area, Geometry.IntSize(1, 1), .1, context, Geometry.FloatPoint(.5, .5))
            first = capture()
            info = first.metadata['kikuchi_simulation']
            self.assertGreater(info['band_count'], 0)
            self.assertGreater(info['thickness_nm'], 0)
            np.testing.assert_allclose(info['crystal_001_direction_xyz'], self.sample.model_info['base_normal_unit_xyz'])
            spectrum = eels.get_frame_data(ea, Geometry.IntSize(1, 1), .1, context, Geometry.FloatPoint(.5, .5))
            self.assertAlmostEqual(spectrum.metadata['eels_simulation']['thickness_nm'], info['thickness_nm']+5.)
            manager.set_value_2d('stage_tilt_rad', Geometry.FloatPoint(x=.01))
            self.assertGreater(float(np.max(abs(capture().data-first.data))), 1.)
            generator = InstrumentDevice.ScanDataGenerator(sample_index=8)
            parameters = ScanDevice.ScanFrameParameters(pixel_size=(64, 64), fov_nm=800., pixel_time_us=1.)
            image = generator.generate_scan_data(camera.instrument, parameters)
            self.assertTrue(np.isfinite(image).all())
            self.assertGreater(float(image.std()), 0.)
            # Far from the prism, carbon attenuation must not spawn crystal lines.
            manager.set_value_2d('stage_position_m', Geometry.FloatPoint())
            self.assertEqual(capture().metadata['kikuchi_simulation']['band_count'], 0)
        finally:
            eels.close()
            camera.close()

    def test_selecting_sample_centers_all_three_scan_profiles(self):
        setup = TestContext.TestSetup()
        with AcquisitionTestContext.AcquisitionTestContext(DeviceConfiguration.AcquisitionContextConfiguration()) as context:
            generator = context.instrument.scan_data_generator
            generator.sample_index = 8
            self.assertEqual(generator.sample.title, 'SingleCathodeOnCarbon')
            stage, fov = generator.sample.initial_view
            self.assertEqual(context.instrument.stage_position_m, stage)
            for index in range(3):
                parameters = context.scan_hardware_source.get_frame_parameters(index)
                self.assertEqual(parameters.fov_nm, fov)
                self.assertEqual(parameters.center_nm, Geometry.FloatPoint())
            position = -np.array([stage.x, stage.y])*1e9
            np.testing.assert_allclose(position, generator.sample.center_nm[:2])
        setup = None

    def test_stage_tilt_changes_shape_conserves_volume_and_updates_camera_geometry(self):
        fixture = KikuchiModel_test.TestKikuchiModel()
        fixture.sample = self.sample
        manager, camera, context, area = fixture.make_camera(128)
        manager.set_value_2d('stage_position_m', self.sample.initial_view[0])
        manager.set_value('C10Control', 4000e-9)
        maps, frames, scans = [], [], []
        try:
            for tilt in (Geometry.FloatPoint(), Geometry.FloatPoint(x=np.radians(20)), Geometry.FloatPoint(y=np.radians(20))):
                manager.set_value_2d('stage_tilt_rad', tilt)
                # Isolate the real-space contribution from the diffraction lines.
                with mock.patch.object(camera, '_apply_kikuchi', side_effect=lambda data, *args: (data, {})):
                    frames.append(camera.get_frame_data(area, Geometry.IntSize(1, 1), .1, context, Geometry.FloatPoint(.5, .5)).data.copy())
                shape = Geometry.IntSize(256, 256)
                _, _, lo, hi = self.sample._maps(self.sample.initial_view[0], Geometry.FloatSize(800, 800), Geometry.FloatPoint(), Geometry.FloatPoint(), shape)
                thickness = self.sample._thickness(lo, hi)
                maps.append(thickness)
                volume = float(thickness.sum(dtype=np.float64))*(800/256)**2
                self.assertAlmostEqual(volume/self.sample.model_info['models']['hexagonal_prism.stl']['volume_nm3'], 1., delta=.003)
                center = Geometry.FloatPoint(x=self.sample.center_nm[0]*1e-9, y=self.sample.center_nm[1]*1e-9)
                layers = self.sample.eels_layers_at(center)
                expected_film = 5./(np.cos(tilt.x)*np.cos(tilt.y))
                self.assertAlmostEqual(layers[0].thickness_nm, expected_film, delta=.002)
                self.assertEqual(layers[0].material.edges[0][0], 284.)
                # Render the actual HAADF path with fixed detector noise.
                generator = InstrumentDevice.ScanDataGenerator(sample_index=8)
                parameters = ScanDevice.ScanFrameParameters(pixel_size=(64, 64), fov_nm=800., pixel_time_us=1.)
                manager.set_value('C10Control', self.sample.center_nm[2]*1e-9)
                with mock.patch('nion.usim_device.Noise.HAADFNoise.apply', side_effect=lambda data, *args: data):
                    haadf = generator.generate_scan_data(camera.instrument, parameters)
                manager.set_value('C10Control', 4000e-9)
                self.assertTrue(np.isfinite(haadf).all())
                scans.append(haadf)
            self.assertGreater(np.count_nonzero((maps[0] > 0) != (maps[1] > 0)), 100)
            self.assertGreater(np.count_nonzero((maps[0] > 0) != (maps[2] > 0)), 100)
            self.assertGreater(float(np.max(abs(frames[0]-frames[1]))), 1.)
            self.assertGreater(float(np.max(abs(frames[0]-frames[2]))), 1.)
            self.assertGreater(float(np.max(abs(scans[0]-scans[1]))), .1)
            self.assertGreater(float(np.max(abs(scans[0]-scans[2]))), .1)
        finally:
            camera.close()
            self.sample.set_stage_tilt(Geometry.FloatPoint())
