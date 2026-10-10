"""Stage Z UI units and camera display coordinates remain consistent."""
import unittest
from unittest import mock
import numpy as np
from nion.utils import Geometry
from nion.ui import TestUI, UserInterface
from nion.swift.test import TestContext
from nion.instrumentation.test import AcquisitionTestContext
from nion.usim_device import DeviceConfiguration
from nionswift_plugin.usim import InstrumentPanel
from nionswift_plugin.usim.test import KikuchiModel_test


class TestDisplayOrientation(unittest.TestCase):
    def test_stage_z_field_uses_micrometres_in_both_directions(self):
        setup = TestContext.TestSetup()
        with AcquisitionTestContext.AcquisitionTestContext(DeviceConfiguration.AcquisitionContextConfiguration()) as context:
            widget = InstrumentPanel.InstrumentWidget(TestUI.UserInterface(), context.instrument)
            try:
                rows = widget.content_widget._contained_widgets
                row = next(row for row in rows if any(isinstance(child, UserInterface.LabelWidget) and child.text == 'Stage Z'
                    for child in getattr(row, '_contained_widgets', [])))
                field = next(child for child in row._contained_widgets if isinstance(child, UserInterface.LineEditWidget))
                context.instrument.SetVal('stage_z_m', 100e-6)
                self.assertIn('100', field.text)
                self.assertIn('µm', field.text)
                field._behavior.on_editing_finished('125 µm')
                self.assertAlmostEqual(context.instrument.GetVal('stage_z_m'), 125e-6)
            finally:
                widget.close()

    def test_full_frame_rotation_and_angular_calibration_match(self):
        fixture = KikuchiModel_test.TestKikuchiModel()
        fixture.setUp()
        manager, camera, context, area = fixture.make_camera(32)
        raw = np.arange(1024, dtype=np.float32).reshape(32, 32)
        try:
            with mock.patch.object(camera, '_apply_kikuchi', return_value=(raw.copy(), {'model': 'orientation_test'})):
                frame = camera.get_frame_data(area, Geometry.IntSize(1, 1), .1, context, Geometry.FloatPoint(.5, .5))
            scale = camera.get_total_counts(.1)/(raw.size*100)
            np.testing.assert_allclose(frame.data, raw[::-1, ::-1]*scale)
            self.assertEqual(frame.metadata['kikuchi_simulation']['display_rotation_deg'], 180)
            native = camera._raw_dimensional_calibrations(area, Geometry.IntSize(1, 1))
            for original, displayed in zip(native, frame.dimensional_calibrations):
                np.testing.assert_allclose(displayed.offset+np.arange(32)*displayed.scale,
                    original.offset+np.arange(31, -1, -1)*original.scale)
        finally:
            camera.close()
