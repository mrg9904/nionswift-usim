import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from nion.instrumentation import stem_controller
from nion.usim_device import InstrumentDevice, KikuchiModel, RonchigramCameraSimulator, SampleSimulator
from nion.utils import Event, Geometry


class TestKikuchiModel(unittest.TestCase):
    def setUp(self):
        self.sample = SampleSimulator.SphericalParticleSample(1000)
        self.crystal = KikuchiModel.load_crystal(self.sample.crystal_cif_path)

    def test_cif_symmetry_mixed_occupancy_and_packaged_copy(self):
        source = Path(__file__).with_name('NNMTO_pristine.cif')
        self.assertEqual(source.read_bytes(), Path(self.sample.crystal_cif_path).read_bytes())
        self.assertEqual(len(self.crystal.elements), 18)
        self.assertEqual(self.crystal.elements.count('O'), 6)
        self.assertAlmostEqual(self.crystal.occupancies[self.crystal.elements.index('Mn')], .29531)
        # R centering: -h+k+l must be divisible by 3.
        self.assertLess(abs(self.crystal.structure_factor((1, 0, 0))), 1e-10)
        self.assertGreater(abs(self.crystal.structure_factor((1, 1, 0))), 1)

    def test_nonorthogonal_metric_and_initial_zone(self):
        a, b, c = self.crystal.cell_angstrom.T
        self.assertAlmostEqual(np.dot(a, b)/np.linalg.norm(a)/np.linalg.norm(b), -.5)
        np.testing.assert_allclose(self.crystal.cell_angstrom.T @ self.crystal.reciprocal, np.eye(3), atol=1e-14)
        orientation = KikuchiModel.orientation_matrix(self.crystal)
        np.testing.assert_allclose(orientation @ c / np.linalg.norm(c), [0, 0, 1], atol=1e-14)
        np.testing.assert_allclose(orientation @ a / np.linalg.norm(a), [1, 0, 0], atol=1e-14)
        # A general [uvw] property is a real-space direction, not an hkl normal.
        direction = self.crystal.cell_angstrom @ [1, 1, 0]
        np.testing.assert_allclose(KikuchiModel.orientation_matrix(self.crystal, (1, 1, 0)) @ direction/np.linalg.norm(direction), [0, 0, 1], atol=1e-14)

    def test_wavelength_bragg_pair_width_and_tilt_sign(self):
        self.assertAlmostEqual(KikuchiModel.wavelength_angstrom(100000), .0370143661, places=9)
        normal = np.array([1., 0, 0])
        theta = math.asin(KikuchiModel.wavelength_angstrom(100000)/(2*2))
        band = KikuchiModel.Band((1, 0, 0), normal, 2., theta, 1.)
        x = np.linspace(-.04, .04, 2001)
        image = KikuchiModel.render_lines([band], x, np.array([0.]))[0]
        self.assertAlmostEqual(x[image.argmax()]-x[image.argmin()], 2*theta, delta=.00008)
        self.assertLess(KikuchiModel.wavelength_angstrom(200000), KikuchiModel.wavelength_angstrom(100000))
        tx = KikuchiModel.orientation_matrix(self.crystal, tx_rad=.02)
        ty = KikuchiModel.orientation_matrix(self.crystal, ty_rad=.02)
        np.testing.assert_allclose(tx @ [0, 0, 1], [0, -math.sin(.02), math.cos(.02)], atol=1e-14)
        np.testing.assert_allclose(ty @ [0, 0, 1], [math.sin(.02), 0, math.cos(.02)], atol=1e-14)
        for voltage in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                KikuchiModel.wavelength_angstrom(voltage)

    def make_camera(self, size=128):
        manager = InstrumentDevice.ValueManager()
        instrument = SimpleNamespace(value_manager=manager, scan_data_generator=SimpleNamespace(sample=self.sample),
            scan_controller=SimpleNamespace(scan_device=SimpleNamespace()), probe_state='parked',
            probe_state_changed_event=Event.Event(), GetVal=manager.get_value, GetVal2D=manager.get_value_2d,
            max_defocus=5000e-9, stage_size_nm=1000, defocus_m=0.)
        # Current scope queries stage coordinates dynamically.
        class Scope(SimpleNamespace):
            @property
            def stage_position_m(self):
                return self.value_manager.stage_position_m
        instrument = Scope(**instrument.__dict__)
        camera = RonchigramCameraSimulator.RonchigramCameraSimulator(instrument, Geometry.IntSize(size, size), 10, 1000)
        camera.noise.enabled = False
        context = stem_controller.ScanContext(Geometry.IntSize(64, 64), Geometry.FloatPoint(), 200., 0.)
        area = Geometry.IntRect(origin=Geometry.IntPoint(), size=Geometry.IntSize(size, size))
        return manager, camera, context, area

    def test_live_camera_probe_tilt_voltage_convergence_and_blanking(self):
        manager, camera, context, area = self.make_camera()
        def capture(position=Geometry.FloatPoint(.5, .5), exposure=.01):
            return camera.get_frame_data(area, Geometry.IntSize(1, 1), exposure, context, position)
        try:
            center = capture()
            info = center.metadata['kikuchi_simulation']
            self.assertEqual(info['zone_axis'], [0, 0, 1])
            self.assertAlmostEqual(info['thickness_nm'], 100)
            self.assertGreater(info['band_count'], 3)
            np.testing.assert_allclose(capture(exposure=.02).data, center.data*2, rtol=2e-7)
            for axis in ('x', 'y'):
                manager.set_value_2d('stage_tilt_rad', Geometry.FloatPoint(**{axis: .01}))
                tilted = capture()
                self.assertGreater(np.max(np.abs(tilted.data-center.data)), 1)
            manager.set_value_2d('stage_tilt_rad', Geometry.FloatPoint())
            np.testing.assert_allclose(capture().data, center.data)
            edge = capture(Geometry.FloatPoint(y=.5, x=.745))
            self.assertAlmostEqual(edge.metadata['kikuchi_simulation']['thickness_nm'], 2*math.sqrt(50**2-49**2))
            vacuum = capture(Geometry.FloatPoint(y=.5, x=.8))
            self.assertEqual(vacuum.metadata['kikuchi_simulation']['band_count'], 0)
            self.assertEqual(vacuum.metadata['kikuchi_simulation']['thickness_nm'], 0)
            manager.set_value('EHT', 200000)
            self.assertTrue(camera._needs_recalculation)
            self.assertGreater(np.max(np.abs(capture().data-center.data)), 1)
            manager.set_value('ConvergenceAngle', .02)
            self.assertEqual(capture().metadata['kikuchi_simulation']['convergence_semiangle_rad'], .02)
            self.sample.zone_axis = (1, 0, 0)
            self.assertEqual(capture().metadata['kikuchi_simulation']['zone_axis'], [1, 0, 0])
            manager.is_blanked = True
            self.assertEqual(capture().data.max(), 0)
        finally:
            camera.close()

    def test_detector_calibration_binned_crop_and_stage_controls(self):
        manager, camera, context, area = self.make_camera()
        try:
            changes = []
            listener = manager.property_changed_event.listen(changes.append)
            manager.set_value('stage_tilt_rad.x', math.radians(1))
            self.assertAlmostEqual(manager.get_value_2d('stage_tilt_rad').x, math.radians(1))
            self.assertIn('stage_tilt_rad', changes)
            listener.close()
            cropped = Geometry.IntRect(origin=Geometry.IntPoint(16, 32), size=Geometry.IntSize(64, 64))
            cal = camera.get_dimensional_calibrations(cropped, Geometry.IntSize(2, 4))
            angle = camera._tv_pixel_angle
            self.assertAlmostEqual(cal[0].offset, (16+.5-64)*angle)
            self.assertAlmostEqual(cal[1].offset, (32+1.5-64)*angle)
            self.assertAlmostEqual(cal[0].scale, 2*angle)
            self.assertAlmostEqual(cal[1].scale, 4*angle)
        finally:
            camera.close()

    def test_tilt_widget_degree_binding_in_both_directions(self):
        import asyncio
        from nion.ui import TestUI, UserInterface
        from nionswift_plugin.usim import InstrumentPanel
        # Earlier acquisition tests close their event loops. Give this UI
        # binding test its own loop instead of relying on process test order.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        manager = InstrumentDevice.ValueManager()
        widget = InstrumentPanel.PositionWidget(TestUI.UserInterface(), 'Stage tilt', manager,
            'stage_tilt_rad', 'deg', 180/math.pi, ('TX', 'TY'))
        fields = [child for child in widget.content_widget._contained_widgets
                  if isinstance(child, UserInterface.LineEditWidget)]
        labels = [child.text for child in widget.content_widget._contained_widgets
                  if isinstance(child, UserInterface.LabelWidget)]
        try:
            self.assertIn('TX', labels)
            self.assertIn('TY', labels)
            self.assertEqual(len(fields), 2)
            fields[0]._behavior.on_editing_finished('1.5 deg')
            fields[1]._behavior.on_editing_finished('-2 deg')
            self.assertAlmostEqual(manager.get_value_2d('stage_tilt_rad').x, math.radians(1.5))
            self.assertAlmostEqual(manager.get_value_2d('stage_tilt_rad').y, math.radians(-2))
            manager.set_value_2d('stage_tilt_rad', Geometry.FloatPoint(x=math.radians(3), y=math.radians(4)))
            self.assertIn('3', fields[0].text)
            self.assertIn('4', fields[1].text)
        finally:
            widget.close()
            loop.close()
            asyncio.set_event_loop(None)

    def test_high_order_family_from_singlecrystal_comparison_is_retained(self):
        old = KikuchiModel.bands(self.crystal, 100000, max_angle_rad=.04, d_min_angstrom=1.)
        expanded = KikuchiModel.bands(self.crystal, 100000, max_angle_rad=.04)
        self.assertEqual(len(old), 3)
        self.assertEqual(len(expanded), 6)
        self.assertIn((3, 0, 0), [b.hkl for b in expanded])
        self.assertTrue(any(b.d_angstrom < 1 for b in expanded))

    def test_defocused_particle_lines_extend_past_uninserted_aperture(self):
        from unittest.mock import patch
        manager, camera, context, area = self.make_camera(256)
        manager.set_value('C10', 500e-9)
        def capture():
            return camera.get_frame_data(area, Geometry.IntSize(1, 1), .01, context, Geometry.FloatPoint(.5, .5))
        try:
            with patch.object(KikuchiModel, 'thickness_contrast', return_value=0):
                base = capture().data.copy()
            camera._needs_recalculation = True
            open_frame = capture().data.copy()
            cal = camera.get_dimensional_calibrations(area, Geometry.IntSize(1, 1))
            y = cal[0].offset + np.arange(256)*cal[0].scale
            x = cal[1].offset + np.arange(256)*cal[1].scale
            outside = np.hypot(y[:, None], x[None, :]) > manager.get_value('ConvergenceAngle')+.003
            self.assertGreater(np.max(np.abs(open_frame-base)[outside]), 1)
            manager.set_value('ConvergenceAngle', .02)
            np.testing.assert_allclose(capture().data, open_frame)
            manager.set_value('ConvergenceAngle', 0.)
            np.testing.assert_allclose(capture().data, open_frame)
            manager.set_value('ConvergenceAngle', .02)
            manager.set_value('S_VOA', 1)
            self.assertEqual(capture().data[outside].max(), 0)
            manager.set_value('S_VOA', 0)
            manager.set_value('S_MOA', 1)
            self.assertEqual(capture().data[outside].max(), 0)
            manager.set_value('S_MOA', 0)
            np.testing.assert_allclose(capture().data, open_frame)
        finally:
            camera.close()

    def test_fresh_electron_noise_on_specimen_and_vacuum_frames(self):
        from nion.usim_device import Noise
        manager, camera, context, area = self.make_camera(128)
        camera.noise = Noise.ElectronCountingNoise(10, seed=123)
        try:
            for position in (Geometry.FloatPoint(.5, .5), Geometry.FloatPoint(.5, .8)):
                one = camera.get_frame_data(area, Geometry.IntSize(1, 1), .01, context, position)
                two = camera.get_frame_data(area, Geometry.IntSize(1, 1), .01, context, position)
                delta = two.data-one.data
                self.assertGreater(np.std(delta), 1)
                # Difference of two independent Poisson frames has variance
                # 2*mean_counts*gain, even with a nonuniform Kikuchi signal.
                predicted = 2*one.data.mean()*10
                self.assertAlmostEqual(float(delta.var()/predicted), 1., delta=.08)
                self.assertEqual(one.metadata, two.metadata)
                self.assertGreaterEqual(one.data.min(), 0)
            manager.set_value('BeamCurrent', 0)
            zero = camera.get_frame_data(area, Geometry.IntSize(1, 1), .01, context, Geometry.FloatPoint(.5, .5))
            self.assertEqual(zero.data.max(), 0)
        finally:
            camera.close()

    def test_thick_specimen_lines_broaden_and_lose_peak_contrast(self):
        x = np.linspace(-.012, .012, 2049)
        pixel_angle = x[1]-x[0]
        line = np.tile(np.exp(-.5*(x/.00035)**2), (8, 1))
        peaks, widths = [], []
        for thickness in (20., 50., 100., 200.):
            real = np.full(line.shape, 100*KikuchiModel.transmission(thickness))
            result, info = KikuchiModel.compose_contrast(real, 100., line, (pixel_angle, pixel_angle))
            background, _ = KikuchiModel.compose_contrast(real, 100., line*0, (pixel_angle, pixel_angle))
            profile = (result-background)[4]
            peaks.append(profile.max())
            widths.append(np.count_nonzero(profile > profile.max()/2)*pixel_angle)
            self.assertEqual(int(profile.argmax()), len(x)//2)
            self.assertAlmostEqual(info['projected_thickness_range_nm'][1], thickness)
        self.assertTrue(np.all(np.diff(widths) > 0), widths)
        self.assertTrue(np.all(np.diff(peaks) < 0), peaks)
        self.assertGreater(widths[-1], 2*widths[0])

    def test_real_space_particle_contrast_and_diffraction_coexist(self):
        from unittest.mock import patch
        manager, camera, context, area = self.make_camera(256)
        manager.set_value('C10', 1000e-9)
        def capture():
            return camera.get_frame_data(area, Geometry.IntSize(1, 1), .01, context, Geometry.FloatPoint(.5, .5))
        try:
            with patch.object(KikuchiModel, 'thickness_contrast', return_value=0):
                real_and_diffuse = capture().data.copy()
            camera._needs_recalculation = True
            mixed = capture()
            self.assertGreater(np.ptp(real_and_diffuse), .15*real_and_diffuse.max())
            self.assertGreater(np.max(np.abs(mixed.data-real_and_diffuse)), 1)
            # Different points in one projected sphere have different path
            # lengths; the compositor must retain the thickness gradient.
            projected = mixed.metadata['kikuchi_simulation']['projected_thickness_range_nm']
            self.assertLess(projected[0], 1)
            self.assertGreater(projected[1], 95)
            self.assertGreater(mixed.metadata['kikuchi_simulation']['probe_diffusion_sigma_rad'], 0)
        finally:
            camera.close()

    def test_diffuse_mixing_preserves_vacuum_and_nonnegative_signal(self):
        pattern = np.random.default_rng(1).uniform(-1, 1, (64, 64))
        vacuum = np.full((64, 64), 100.)
        result, info = KikuchiModel.compose_contrast(vacuum, 100, pattern, (.0004, .0004))
        np.testing.assert_allclose(result, vacuum)
        self.assertEqual(info['diffuse_fraction_range'], [0, 0])
        dark, _ = KikuchiModel.compose_contrast(vacuum*0, 100, pattern, (.0004, .0004))
        self.assertEqual(dark.max(), 0)
        thick = np.full((64, 64), 100*KikuchiModel.transmission(300))
        result, _ = KikuchiModel.compose_contrast(thick, 100, pattern, (.0004, .0004))
        self.assertGreaterEqual(result.min(), 0)

    def test_beam_centre_in_vacuum_still_illuminates_defocused_crystal(self):
        from unittest.mock import patch
        manager, camera, context, area = self.make_camera(256)
        manager.set_value('C10', 1000e-9)
        position = Geometry.FloatPoint(.5, .8)  # x=60 nm, sphere radius=50 nm
        def capture():
            return camera.get_frame_data(area, Geometry.IntSize(1, 1), .01, context, position)
        try:
            with patch.object(KikuchiModel, 'thickness_contrast', return_value=0):
                baseline = capture().data.copy()
            camera._needs_recalculation = True
            result = capture()
            info = result.metadata['kikuchi_simulation']
            self.assertEqual(info['thickness_nm'], 0)  # central ray misses
            self.assertGreater(info['band_count'], 0)
            self.assertGreater(info['projected_thickness_range_nm'][1], 90)
            self.assertGreater(np.max(np.abs(result.data-baseline)), 1)
            # Move the full projected particle out of the camera field.
            manager.set_value_2d('stage_position_m', Geometry.FloatPoint(x=-5000e-9))
            self.assertEqual(capture().metadata['kikuchi_simulation']['band_count'], 0)
        finally:
            camera.close()


if __name__ == '__main__':
    unittest.main()
