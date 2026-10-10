"""Exercise microscope controls from scan and Ronchigram display events."""
import math
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np
from nion.swift import DisplayPanel
from nion.device_kit import CameraDevice
from nion.ui import CanvasItem, TestUI
from nion.utils import Geometry
from nionswift_plugin.usim import InteractiveControls, InteractiveControlSettings
from nionswift_plugin.usim.test import KikuchiModel_test


class TestInteractiveControls(unittest.TestCase):
    def setUp(self):
        fixture = KikuchiModel_test.TestKikuchiModel()
        fixture.setUp()
        self.manager, self.camera, self.context, self.area = fixture.make_camera()
        self.instrument = self.camera.instrument
        self.instrument.SetVal = self.manager.set_value
        self.instrument.get_value_2d = self.manager.get_value_2d
        self.instrument.set_value_2d = self.manager.set_value_2d
        self.manager.ronchigram_camera = SimpleNamespace(camera_id='usim_ronchigram_camera',
            simulator=self.camera)
        self.parameters = SimpleNamespace(fov_nm=200., fov_size_nm=Geometry.FloatSize(200, 200), rotation_rad=0.)
        self.scan = SimpleNamespace(hardware_source_id='usim_scan', selected_profile_index=0,
            get_frame_parameters=lambda _: self.parameters,
            set_frame_parameters=lambda _, parameters: setattr(self, 'parameters', parameters))
        self.controls = InteractiveControls.InteractiveControlManager(self.instrument, SimpleNamespace(hardware_source=self.scan))

    def tearDown(self):
        self.controls.close()
        self.camera.close()

    def panel(self, source):
        return SimpleNamespace(data_item=SimpleNamespace(metadata={'hardware_source': {'hardware_source_id': source}},
            data_shape=(128, 128)), tool_mode='pointer')

    def key(self, text, shift=False, control=False):
        return TestUI.Key(text, text, CanvasItem.KeyboardModifiers(shift=shift, control=control))

    def capture(self):
        return self.camera.get_frame_data(self.area, Geometry.IntSize(1, 1), .1, self.context, Geometry.FloatPoint(.5, .5))

    def test_focus_is_unlimited_and_continues_from_manual_raw_or_effective_value(self):
        panel = self.panel('usim_scan')
        self.manager.set_value('C10', 20000e-9)
        self.assertTrue(self.controls.handle_key_pressed(panel, self.key('f')))
        self.assertAlmostEqual(self.manager.get_value('C10'), 20010e-9)
        self.manager.set_value('C10Control', -30000e-9)
        self.assertTrue(self.controls.handle_key_pressed(panel, self.key('d')))
        self.assertAlmostEqual(self.manager.get_value('C10Control'), -30010e-9)
        self.assertTrue(self.controls.handle_mouse_wheel(SimpleNamespace(delegate=panel), 0, 120, False))
        self.assertAlmostEqual(self.manager.get_value('C10Control'), -30009e-9)

    def test_brightness_contrast_shift_reverse_and_other_shortcuts_are_ignored(self):
        from nionswift_plugin.usim import ScanProfileControls
        for source in ('usim_scan', 'usim_ronchigram_camera'):
            panel = self.panel(source)
            brightness, contrast = ScanProfileControls.get_tone(self.parameters)
            self.assertTrue(self.controls.handle_key_pressed(panel, self.key('b')))
            np.testing.assert_allclose(ScanProfileControls.get_tone(self.parameters), (brightness+.05, contrast), atol=1e-12)
            self.assertTrue(self.controls.handle_key_pressed(panel, self.key('c')))
            after_brightness, after_contrast = ScanProfileControls.get_tone(self.parameters)
            self.assertAlmostEqual(after_brightness, brightness+.05)
            self.assertAlmostEqual(after_contrast, contrast*1.1)
            self.assertTrue(self.controls.handle_key_pressed(panel, self.key('B', shift=True)))
            self.assertTrue(self.controls.handle_key_pressed(panel, self.key('C', shift=True)))
            np.testing.assert_allclose(ScanProfileControls.get_tone(self.parameters), (brightness, contrast), atol=1e-12)
            self.assertFalse(self.controls.handle_key_pressed(panel, self.key('b', control=True)))
        self.assertFalse(self.controls.handle_key_pressed(self.panel('usim_eels_camera'), self.key('b')))

    def test_focus_wheel_and_keys_in_both_displays_and_unrelated_data_ignored(self):
        for source in ('usim_scan', 'usim_ronchigram_camera'):
            panel = self.panel(source)
            canvas = SimpleNamespace(delegate=panel)
            initial = self.manager.get_value('C10')
            self.assertTrue(self.controls.handle_key_pressed(panel, self.key('f')))
            self.assertAlmostEqual(self.manager.get_value('C10'), initial+10e-9)
            self.assertTrue(self.controls.handle_mouse_wheel(canvas, 0, 120, False))
            self.assertAlmostEqual(self.manager.get_value('C10'), initial+11e-9)
            self.assertTrue(self.controls.handle_key_pressed(panel, self.key('d')))
            self.assertTrue(self.controls.handle_mouse_wheel(canvas, 0, -120, False))
            self.assertAlmostEqual(self.manager.get_value('C10'), initial)
            self.assertFalse(self.controls.handle_mouse_wheel(canvas, 10, 0, True))
        for source in ('other_camera', 'usim_eels_camera'):
            panel = self.panel(source)
            self.assertFalse(self.controls.handle_key_pressed(panel, self.key('f')))
            self.assertFalse(self.controls.handle_mouse_wheel(SimpleNamespace(delegate=panel), 0, 120, False))

    def test_installed_keyboard_hook_routes_selected_display_and_restores_nion_handler(self):
        with mock.patch.object(DisplayPanel.DisplayPanel, '_handle_key_pressed', return_value=False) as original:
            InteractiveControls.run(self.instrument, SimpleNamespace(hardware_source=self.scan))
            try:
                for source in ('usim_scan', 'usim_ronchigram_camera'):
                    panel = self.panel(source)
                    self.assertTrue(DisplayPanel.DisplayPanel._handle_key_pressed(panel, self.key('right')))
                    self.assertTrue(DisplayPanel.DisplayPanel._handle_key_pressed(panel, self.key('f')))
                original.assert_not_called()
                self.assertFalse(DisplayPanel.DisplayPanel._handle_key_pressed(self.panel('other_camera'), self.key('right')))
                original.assert_called_once()
            finally:
                InteractiveControls.stop()
            self.assertIs(DisplayPanel.DisplayPanel._handle_key_pressed, original)

    def test_arrow_keys_change_separate_tilt_axes_in_both_displays(self):
        for source in ('usim_scan', 'usim_ronchigram_camera'):
            panel = self.panel(source)
            self.manager.set_value_2d('stage_tilt_rad', Geometry.FloatPoint())
            baseline = self.capture()
            self.assertTrue(self.controls.handle_key_pressed(panel, self.key('right')))
            self.assertAlmostEqual(self.manager.get_value_2d('stage_tilt_rad').y, math.radians(.1))
            self.assertEqual(self.manager.get_value_2d('stage_tilt_rad').x, 0)
            tilted = self.capture()
            self.assertGreater(np.max(np.abs(tilted.data-baseline.data)), 1)
            self.assertTrue(self.controls.handle_key_pressed(panel, self.key('up')))
            self.assertAlmostEqual(self.manager.get_value_2d('stage_tilt_rad').x, math.radians(.1))
            self.assertTrue(self.controls.handle_key_pressed(panel, self.key('left')))
            self.assertTrue(self.controls.handle_key_pressed(panel, self.key('down')))
            self.assertEqual(self.manager.get_value_2d('stage_tilt_rad'), Geometry.FloatPoint())
            self.assertFalse(self.controls.handle_key_pressed(panel, self.key('up', shift=True)))
            self.assertFalse(self.controls.handle_key_pressed(panel, self.key('t')))
            self.assertFalse(self.controls.handle_key_pressed(panel, self.key('y')))
            self.assertFalse(self.controls.handle_key_pressed(panel, self.key('f', shift=True)))

    def test_coarse_and_fine_focus_tilt_and_fov_in_both_displays(self):
        for source in ('usim_scan', 'usim_ronchigram_camera'):
            panel = self.panel(source)
            for shift, multiplier in ((False, 10.), (True, .1)):
                initial_focus = self.manager.get_value('C10')
                self.assertFalse(self.controls.handle_key_pressed(panel, self.key('f', shift=shift, control=True)))
                self.assertFalse(self.controls.handle_key_pressed(panel, self.key('d', shift=shift, control=True)))
                self.assertAlmostEqual(self.manager.get_value('C10'), initial_focus)
                self.manager.set_value_2d('stage_tilt_rad', Geometry.FloatPoint())
                for direction, axis, sign in (('up', 'x', 1), ('down', 'x', -1), ('left', 'y', -1), ('right', 'y', 1)):
                    before = self.manager.get_value_2d('stage_tilt_rad')
                    self.assertTrue(self.controls.handle_key_pressed(panel, self.key(direction, shift=shift, control=True)))
                    after = self.manager.get_value_2d('stage_tilt_rad')
                    self.assertAlmostEqual(getattr(after, axis)-getattr(before, axis), sign*math.radians(.1)*multiplier)
                self.parameters.fov_nm = 10000.
                self.assertTrue(self.controls.handle_key_pressed(panel, self.key('r', shift=shift, control=True)))
                self.assertAlmostEqual(self.parameters.fov_nm, 10000.*.8**multiplier)
                self.assertTrue(self.controls.handle_key_pressed(panel, self.key('e', shift=shift, control=True)))
                self.assertAlmostEqual(self.parameters.fov_nm, 10000.)
        # Real Qt Ctrl+F can have non-printing text; use its physical key code.
        class PhysicalKey(TestUI.Key):
            @property
            def key(self):
                return ord('F')
        key = PhysicalKey('\x06', 'f', CanvasItem.KeyboardModifiers(control=True))
        before = self.manager.get_value('C10')
        self.assertFalse(self.controls.handle_key_pressed(self.panel('usim_scan'), key))
        self.assertAlmostEqual(self.manager.get_value('C10'), before)
        self.assertFalse(self.controls.handle_key_pressed(self.panel('usim_eels_camera'), key))

    def test_ronchigram_double_click_uses_aberration_mapping_and_defocus_sign(self):
        panel = self.panel('usim_ronchigram_camera')
        # Camera frames can also carry the originating scan context. Camera
        # identity must win so angular pixels never use the HAADF scan FoV.
        panel.data_item.metadata['scan'] = {'hardware_source_id': 'usim_scan'}
        point = Geometry.FloatPoint(63.5, 95.5)
        canvas = SimpleNamespace(delegate=panel, mouse_mapping=SimpleNamespace(map_point_widget_to_image=lambda _: point))
        deltas = []
        for focus in (1000e-9, -1000e-9):
            self.manager.set_value_2d('stage_position_m', Geometry.FloatPoint())
            self.manager.set_value('C10Control', focus)
            self.capture()
            delta = self.camera.stage_displacement_for_pixel(point, (128, 128))
            self.assertIsNotNone(delta)
            self.assertTrue(self.controls.handle_double_click(canvas, 96, 64, CanvasItem.KeyboardModifiers()))
            np.testing.assert_allclose(tuple(self.manager.get_value_2d('stage_position_m')), tuple(-delta), atol=1e-15)
            deltas.append(delta.x)
        self.assertLess(deltas[0], 0)
        self.assertAlmostEqual(deltas[0], -deltas[1])
        self.manager.set_value('C10Control', 1000e-9)
        self.manager.set_value_2d('C12Control', Geometry.FloatPoint(x=500e-9, y=0))
        self.capture()
        astigmatic = self.camera.stage_displacement_for_pixel(point, (128, 128))
        self.assertAlmostEqual(astigmatic.x, 1.5*deltas[0])
        self.manager.set_value('C10Control', 700e-9)
        self.assertFalse(self.controls.handle_double_click(canvas, 96, 64, CanvasItem.KeyboardModifiers()))
        self.capture()
        self.assertFalse(self.controls.handle_double_click(canvas, 96, 64, CanvasItem.KeyboardModifiers(shift=True)))

    def test_scan_double_click_retains_fov_rotation_and_zoom_works_in_both(self):
        panel = self.panel('usim_scan')
        canvas = SimpleNamespace(delegate=panel, mouse_mapping=SimpleNamespace(
            map_point_widget_to_image=lambda _: Geometry.FloatPoint(63.5, 95.5)))
        self.parameters.rotation_rad = math.pi/2
        self.assertTrue(self.controls.handle_double_click(canvas, 96, 64, CanvasItem.KeyboardModifiers()))
        stage = self.manager.get_value_2d('stage_position_m')
        self.assertAlmostEqual(stage.x, 0, delta=1e-15)
        self.assertAlmostEqual(stage.y, -50e-9)
        for source in ('usim_scan', 'usim_ronchigram_camera'):
            panel = self.panel(source)
            self.assertTrue(self.controls.handle_key_pressed(panel, self.key('r')))
            self.assertEqual(self.parameters.fov_nm, 160)
            self.assertTrue(self.controls.handle_key_pressed(panel, self.key('e')))
            self.assertEqual(self.parameters.fov_nm, 200)

    def test_double_click_with_actual_device_kit_camera(self):
        self.instrument.camera_sensor_dimensions = lambda _: (128, 128)
        self.instrument.camera_readout_area = lambda _: (0, 0, 128, 128)
        device = CameraDevice.Camera('usim_ronchigram_camera', 'ronchigram', 'uSim Ronchigram', self.camera, self.instrument)
        try:
            self.manager.ronchigram_camera = device
            self.assertFalse(hasattr(device, 'stage_displacement_for_pixel'))
            self.manager.set_value('C10Control', 1000e-9)
            self.capture()
            canvas = SimpleNamespace(delegate=self.panel(device.camera_id), mouse_mapping=SimpleNamespace(
                map_point_widget_to_image=lambda _: Geometry.FloatPoint(63.5, 95.5)))
            self.assertTrue(self.controls.handle_double_click(canvas, 96, 64, CanvasItem.KeyboardModifiers()))
            self.assertGreater(self.manager.get_value_2d('stage_position_m').x, 0)
            # A camera without compatible simulation geometry must fall through.
            self.manager.ronchigram_camera = SimpleNamespace(camera_id=device.camera_id)
            self.assertFalse(self.controls.handle_double_click(canvas, 96, 64, CanvasItem.KeyboardModifiers()))
        finally:
            # The real Camera owns/closes the simulator; tearDown closes it once.
            with mock.patch.object(self.camera, 'close'):
                device.close()
