"""Verify the actual Scan Control panel and persistent display profile values."""
import unittest
import numpy as np
from nion.swift.test import TestContext
from nion.swift.model import DataItem
from nion.ui import Declarative
from nion.device_kit import ScanDevice
from nion.instrumentation.test import AcquisitionTestContext
from nion.usim_device import DeviceConfiguration
from nionswift_plugin.nion_instrumentation_ui import ScanControlPanel
from nionswift_plugin.usim import ScanProfileControls


class TestScanProfileControls(unittest.TestCase):
    def setUp(self):
        self.setup = TestContext.TestSetup()
        self.original = ScanControlPanel.ScanPanelController.__init__
        ScanProfileControls.run()

    def tearDown(self):
        ScanProfileControls.stop()
        self.assertIs(ScanControlPanel.ScanPanelController.__init__, self.original)
        self.setup = None

    def test_panel_fields_profiles_and_display_transfer_leave_raw_data_unchanged(self):
        with AcquisitionTestContext.AcquisitionTestContext(DeviceConfiguration.AcquisitionContextConfiguration()) as context:
            source = context.scan_hardware_source
            controller = ScanControlPanel.ScanPanelController(context.document_controller, source)
            self.assertNotIn('@binding(_model.fov_label_color)', str(controller.ui_view))
            self.assertNotIn('@binding(_model.fov_label_tool_tip)', str(controller.ui_view))
            widget = Declarative.DeclarativeWidget(context.document_controller.ui, context.document_controller.event_loop, controller)
            try:
                tone = controller._usim_tone
                tone.brightness_str = '.25'
                tone.contrast_str = '2.0'
                parameters = source.get_frame_parameters(0)
                self.assertEqual(ScanProfileControls.get_tone(parameters), (.25, 2.))
                restored = ScanDevice.ScanFrameParameters(parameters.as_dict())
                self.assertEqual(ScanProfileControls.get_tone(restored), (.25, 2.))
                for hardware_id in (source.hardware_source_id, 'usim_ronchigram_camera', 'usim_eels_camera'):
                    data = np.arange(16, dtype=np.float32).reshape(4, 4)
                    item = DataItem.DataItem(data.copy())
                    item.metadata = {'hardware_source': {'hardware_source_id': hardware_id},
                        'scan': {'hardware_source_id': source.hardware_source_id}}
                    context.document_model.append_data_item(item)
                    context.document_controller.periodic()
                    tone.refresh()
                    channel = context.document_model.get_display_item_for_data_item(item).display_data_channel
                    expected = (0., 1.) if hardware_id == 'usim_eels_camera' else (.25, 2.)
                    self.assertEqual((channel.brightness, channel.contrast), expected)
                    np.testing.assert_array_equal(item.data, data)
                source.set_selected_profile_index(1)
                tone.refresh()
                self.assertEqual((tone.brightness_str, tone.contrast_str), ('0.00', '1.00'))
                tone.contrast_str = '3'
                source.set_selected_profile_index(0)
                tone.refresh()
                self.assertEqual((tone.brightness_str, tone.contrast_str), ('0.25', '2.00'))
                tone.contrast_str = '-1'
                self.assertEqual(tone.contrast_str, '2.00')
            finally:
                widget.close()
