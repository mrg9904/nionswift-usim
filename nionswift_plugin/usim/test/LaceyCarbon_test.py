"""Shared combined-STL geometry in scan, EELS, and Ronchigram paths."""
import unittest
from unittest import mock
import numpy as np
from nion.utils import Geometry
from nion.device_kit import ScanDevice
from nion.usim_device import LaceyCarbonSample, SimulationSettings, InstrumentDevice, EELSCameraSimulator
from nionswift_plugin.usim.test import KikuchiModel_test


class TestLaceyCarbon(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with mock.patch.object(SimulationSettings, 'STL_SURFACE_BACKEND', 'cpu'):
            cls.sample = LaceyCarbonSample.LaceyCarbonSample(1000)
        shape = Geometry.IntSize(128, 128)
        cls.thickness = cls.sample.thickness_projection(Geometry.FloatPoint(),
            Geometry.FloatSize(54000, 54000), Geometry.FloatPoint(), Geometry.FloatPoint(), shape)
        cls.points = {}
        for name, mask in [('carbon', cls.thickness > 0), ('vacuum', cls.thickness == 0)]:
            y, x = np.argwhere(mask)[len(np.argwhere(mask))//2]
            cls.points[name] = Geometry.FloatPoint(y=(-27000+(y+.5)*54000/128)*1e-9,
                                                   x=(-27000+(x+.5)*54000/128)*1e-9)
        cls.points['copper'] = Geometry.FloatPoint(x=40000e-9)

    def test_registration_and_initial_view(self):
        generator = InstrumentDevice.ScanDataGenerator(sample_index=7)
        self.assertEqual(generator.sample_titles[7], 'LaceyCarbon')
        self.assertEqual(generator.sample.initial_view, (Geometry.FloatPoint(), 200.))
        fixture = KikuchiModel_test.TestKikuchiModel()
        fixture.sample = generator.sample
        manager, camera, _, _ = fixture.make_camera(64)
        try:
            parameters = ScanDevice.ScanFrameParameters()
            parameters.size = Geometry.IntSize(64, 64)
            parameters.fov_nm = 100000.
            parameters.pixel_time_us = 1.
            image = generator.generate_scan_data(camera.instrument, parameters)
            self.assertEqual(image.shape, (64, 64))
            self.assertTrue(np.isfinite(image).all())
            self.assertGreater(float(image.std()), 0.)
        finally:
            camera.close()

    def test_shared_geometry_in_all_three_detectors(self):
        fixture = KikuchiModel_test.TestKikuchiModel()
        fixture.sample = self.sample
        manager, ronchi, context, area = fixture.make_camera(64)
        eels = EELSCameraSimulator.EELSCameraSimulator(ronchi.instrument, Geometry.IntSize(8, 2048), 10)
        eels.noise.enabled = False
        ea = Geometry.IntRect(origin=Geometry.IntPoint(), size=Geometry.IntSize(8, 2048))
        try:
            for name, thickness in [('carbon', 5.), ('vacuum', 0.), ('copper', 10000.)]:
                position = self.points[name]
                layers = self.sample.eels_layers_at(position)
                self.assertAlmostEqual(sum(l.thickness_nm for l in layers), thickness)
                if layers:
                    self.assertEqual(layers[0].material.edges[0][0], 284. if name == 'carbon' else 75.)
                offset = Geometry.FloatPoint(y=-position.y, x=-position.x)
                manager.set_value_2d('stage_position_m', offset)
                spectrum = eels.get_frame_data(ea, Geometry.IntSize(1, 1), .01, context, Geometry.FloatPoint(.5, .5))
                self.assertAlmostEqual(spectrum.metadata['eels_simulation']['thickness_nm'], thickness)
                frame = ronchi.get_frame_data(area, Geometry.IntSize(1, 1), .01, context, Geometry.FloatPoint(.5, .5))
                self.assertTrue(np.isfinite(frame.data).all())
                self.assertGreaterEqual(frame.data.min(), 0)
                shape = Geometry.IntSize(1, 1)
                planes = self.sample.generate_depth_planes(offset, Geometry.FloatSize(1, 1),
                    Geometry.FloatPoint(), Geometry.FloatPoint(), shape, 1.)
                intensity = sum(float(data[0, 0]) for _, data in planes)
                self.assertAlmostEqual(intensity, {'carbon': .25, 'vacuum': 0., 'copper': 1.}[name])
            manager.set_value_2d('stage_position_m', Geometry.FloatPoint())
            one = ronchi._source_image(area, Geometry.IntSize(1, 1))
            self.assertIs(one, ronchi._source_image(area, Geometry.IntSize(1, 1)))
        finally:
            eels.close()
            ronchi.close()

    def test_large_defocus_retains_film_instead_of_black_source_boundary(self):
        fixture = KikuchiModel_test.TestKikuchiModel()
        fixture.sample = self.sample
        manager, camera, context, area = fixture.make_camera(128)
        frames = []
        try:
            for focus_nm in (1000., 10000., 50000., -50000.):
                manager.set_value('C10Control', focus_nm * 1e-9)
                frame = camera.get_frame_data(area, Geometry.IntSize(1, 1), .1,
                    context, Geometry.FloatPoint(.5, .5)).data.copy()
                # All rays here stay inside the 54 um opening. Carbon and
                # vacuum transmit; zeros would be the old source-FoV clipping.
                self.assertGreater(float(frame.min()), 0.)
                self.assertGreater(float(np.ptp(frame)), float(frame.max()) * .02)
                frames.append(frame)
            self.assertGreater(float(np.max(abs(frames[2] - frames[1]))), 1.)
            # Clicking a displayed feature must still use the expanded source
            # FoV, rather than treating the entire film as the old 1 um field.
            delta = camera.stage_displacement_for_pixel(Geometry.FloatPoint(.5, .75), (128, 128))
            self.assertIsNotNone(delta)
            self.assertGreater(abs(delta.x), 1e-6)
        finally:
            camera.close()
