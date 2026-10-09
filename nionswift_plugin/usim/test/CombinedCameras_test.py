"""Verify EELS and Ronchigram share one specimen and geometry after integration."""
import math
import unittest
import numpy as np
from nion.usim_device import EELSCameraSimulator
from nion.utils import Geometry
from nionswift_plugin.usim.test import KikuchiModel_test


class TestCombinedCameras(unittest.TestCase):
    def test_shared_sphere_thickness_tilt_and_offaxis_illumination(self):
        fixture = KikuchiModel_test.TestKikuchiModel()
        fixture.setUp()
        manager, ronchigram, context, area = fixture.make_camera()
        eels = EELSCameraSimulator.EELSCameraSimulator(ronchigram.instrument, Geometry.IntSize(8, 1024), 10)
        eels.noise.enabled = False
        eels_area = Geometry.IntRect(origin=Geometry.IntPoint(), size=Geometry.IntSize(8, 1024))
        bins = Geometry.IntSize(1, 1)
        manager.set_value('C10Control', 1000e-9)
        def capture(x):
            probe = Geometry.FloatPoint(.5, x)
            return (eels.get_frame_data(eels_area, bins, .1, context, probe),
                    ronchigram.get_frame_data(area, bins, .1, context, probe))
        try:
            for x, thickness in ((.5, 100), (.745, 2*math.sqrt(50**2-49**2)), (.8, 0)):
                spectrum, diffraction = capture(x)
                eels_info = spectrum.metadata['eels_simulation']
                ronchi_info = diffraction.metadata['kikuchi_simulation']
                self.assertAlmostEqual(eels_info['thickness_nm'], thickness)
                self.assertAlmostEqual(ronchi_info['thickness_nm'], thickness)
                self.assertAlmostEqual(eels_info['background_event_optical_depth'], thickness/100*.25)
                self.assertGreater(ronchi_info['band_count'], 0)  # including off-axis illuminated sphere
            before_eels, before_ronchi = capture(.5)
            manager.set_value_2d('stage_tilt_rad', Geometry.FloatPoint(.01, 0))
            after_eels, after_ronchi = capture(.5)
            # Rotating a sphere changes diffraction orientation, not its chord.
            np.testing.assert_array_equal(after_eels.data, before_eels.data)
            self.assertGreater(np.max(np.abs(after_ronchi.data-before_ronchi.data)), 1)
            manager.set_value_2d('stage_position_m', Geometry.FloatPoint(0, -40e-9))
            spectrum, diffraction = capture(.5)
            self.assertAlmostEqual(spectrum.metadata['eels_simulation']['thickness_nm'], 60)
            self.assertAlmostEqual(diffraction.metadata['kikuchi_simulation']['thickness_nm'], 60)
        finally:
            eels.close()
            ronchigram.close()
