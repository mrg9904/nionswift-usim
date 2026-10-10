import unittest
from unittest import mock
import numpy as np
from nion.utils import Geometry
from nion.usim_device import ThousandCathodeSample, SimulationSettings, SampleGeometry
from nion.usim_device import DeviceConfiguration
from nion.usim_device import HAADFFocusModel
from nion.usim_device import KikuchiModel
from nion.device_kit import ScanDevice
from nion.instrumentation.test import AcquisitionTestContext
from nion.swift.test import TestContext
from nionswift_plugin.usim.test import KikuchiModel_test


class TestThousandCathode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with mock.patch.object(SimulationSettings, 'STL_SURFACE_BACKEND', 'cpu'):
            cls.sample = ThousandCathodeSample.ThousandCathodeSample(1000)

    def setUp(self):
        self.sample.set_stage_tilt(Geometry.FloatPoint())

    def test_geometry_orientation_and_culling(self):
        sample = self.sample
        self.assertEqual(sample.title, '1000CathodeParticleOnCarbon')
        self.assertEqual(len(sample.rotations), 1000)
        for i in (0, 237, 999):
            normal = np.array(sample.model_info['particles'][i]['base_normal_abc'], float)
            np.testing.assert_allclose(sample.rotations[i, :, 2], normal/np.linalg.norm(normal))
            initial = np.array(sample.model_info['particles'][i]['base_normal_abc_initial'], float)
            np.testing.assert_allclose(np.asarray(sample.model_info['initial_rotation_matrix']) @ (initial/np.linalg.norm(initial)),
                                       sample.rotations[i, :, 2])
        offset, _ = sample.initial_view
        args = offset, Geometry.FloatSize(2000, 2000), Geometry.FloatPoint(), Geometry.FloatPoint(), Geometry.IntSize(64, 64)
        with mock.patch.object(sample, 'crystal', wraps=sample.crystal) as project:
            sample._projection_key = None
            sample.thickness_projection(*args)
            self.assertIn(1, sample.last_visible_particle_ids)
            self.assertLess(project.call_count, 30)
            self.assertEqual(project.call_count, len(sample.last_visible_particle_ids))
        tilt = Geometry.FloatPoint(x=.3, y=-.2)
        sample.set_stage_tilt(tilt)
        transformed = (sample.vertices-sample.center_nm) @ SampleGeometry.stage_rotation(tilt).T+sample.center_nm
        np.testing.assert_allclose(sample._bounds[:, 0], transformed.min(axis=1))

    def test_nine_grid_windows_only_center_is_populated(self):
        sample = self.sample
        self.assertEqual(sample.model_info['initial_rotation_deg'], 60.)
        self.assertEqual(sample.model_info['grid_tiles_per_axis'], 3)
        np.testing.assert_allclose(sample.model_info['overall_grid_size_local_nm'], [255000., 255000., 10000.])
        rotation = np.asarray(sample.model_info['initial_rotation_matrix'])
        for center in sample.model_info['grid_tile_centers_local_nm']:
            if not np.any(center):
                continue
            lab = rotation @ center
            position = Geometry.FloatPoint(x=lab[0]*1e-9, y=lab[1]*1e-9)
            self.assertEqual(sample.eels_layers_at(position), [])
            self.assertEqual(len(sample.visible_indices([lab[0]], [lab[1]])), 0)
            copper = rotation @ (np.asarray(center)+[34000., 0., 0.])
            layers = sample.eels_layers_at(Geometry.FloatPoint(x=copper[0]*1e-9, y=copper[1]*1e-9))
            self.assertEqual(len(layers), 1)
            self.assertAlmostEqual(layers[0].thickness_nm, 10000., places=2)

    def test_stacked_material_depths_exclude_vacuum(self):
        sample = self.sample
        record = next(r for r in sample.model_info['particles'] if r['stacked'])
        center = sample.vertices[record['id']-1].mean(axis=0)
        offset = Geometry.FloatPoint(x=-center[0]*1e-9, y=-center[1]*1e-9)
        args = offset, Geometry.FloatSize(1, 1), Geometry.FloatPoint(), Geometry.FloatPoint(), Geometry.IntSize(1, 1)
        expected = sample.thickness_projection(*args)[0, 0]
        planes = sample.generate_depth_planes(*args, 2.)
        self.assertAlmostEqual(sum(float(p[0, 0])*20 for _, p in planes), expected, delta=.001)
        position = Geometry.FloatPoint(x=center[0]*1e-9, y=center[1]*1e-9)
        self.assertAlmostEqual(sum(layer.thickness_nm for layer in sample.eels_layers_at(position)), expected, delta=.001)
        self.assertLessEqual(len(planes), 65)

    def test_ronchigram_uses_individual_visible_orientations(self):
        fixture = KikuchiModel_test.TestKikuchiModel()
        fixture.sample = self.sample
        manager, camera, context, area = fixture.make_camera(128)
        manager.set_value_2d('stage_position_m', self.sample.initial_view[0])
        manager.set_value('C10Control', 20000e-9)
        try:
            with mock.patch.object(SimulationSettings, 'RONCHIGRAM_BACKEND', 'cpu'):
                result = camera.get_frame_data(area, Geometry.IntSize(1, 1), .1, context, Geometry.FloatPoint(.5, .5))
            info = result.metadata['kikuchi_simulation']
            self.assertGreater(len(info['crystals']), 1)
            self.assertLess(len(info['visible_particle_ids']), 1000)
            for crystal in info['crystals']:
                np.testing.assert_allclose(crystal['crystal_rotation_matrix'], self.sample.rotations[crystal['particle_id']-1])
            self.assertTrue(np.isfinite(result.data).all())
            cache = camera._multi_crystal_cache
            total = sum(p[0].nbytes+c.pattern.nbytes+(c.bank.nbytes if c.bank is not None else 0)
                        for _, p, c in cache.values())
            self.assertLessEqual(total, SimulationSettings.RONCHIGRAM_PATTERN_CACHE_BYTES)
        finally:
            camera.close()

    def test_sample_selection_updates_profiles_and_stage(self):
        setup = TestContext.TestSetup()
        with AcquisitionTestContext.AcquisitionTestContext(DeviceConfiguration.AcquisitionContextConfiguration()) as context:
            generator = context.instrument.scan_data_generator
            self.assertIn(9, generator.selectable_sample_indices)
            generator.sample_index = 9
            self.assertEqual(generator.sample.title, '1000CathodeParticleOnCarbon')
            stage, fov = generator.sample.initial_view
            self.assertEqual(context.instrument.stage_position_m, stage)
            for index in range(3):
                self.assertEqual(context.scan_hardware_source.get_frame_parameters(index).fov_nm, fov)
            parameters = ScanDevice.ScanFrameParameters(pixel_size=(64, 64), fov_nm=fov, pixel_time_us=1.)
            image = generator.generate_scan_data(context.instrument, parameters)
            self.assertTrue(np.isfinite(image).all())
            self.assertGreater(float(image.std()), 0.)

    def test_sparse_gpu_depths_match_cpu_focus(self):
        try:
            import cupy as cp
            if not cp.cuda.runtime.getDeviceCount():
                self.skipTest('CUDA unavailable')
        except ImportError:
            self.skipTest('CuPy unavailable')
        sample = self.sample
        args = sample.initial_view[0], Geometry.FloatSize(2000, 2000), Geometry.FloatPoint(), Geometry.FloatPoint(), Geometry.IntSize(64, 64)
        sample._planes_key = None
        with mock.patch.object(SimulationSettings, 'STL_SURFACE_BACKEND', 'cpu'):
            cpu = sample.generate_depth_planes(*args, 2000/64)
        sample._planes_key = None
        with mock.patch.object(SimulationSettings, 'STL_SURFACE_BACKEND', 'gpu'):
            gpu = sample.generate_depth_planes(*args, 2000/64)
        self.assertTrue(getattr(gpu, '_on_gpu', False))
        for focus in (0., 100000e-9):
            settings = dict(defocus_m=focus, best_focus_m=0., convergence_angle_rad=.02,
                            pixel_size_y_nm=2000/64, pixel_size_x_nm=2000/64)
            expected = HAADFFocusModel.apply_depth_planes_defocus(cpu, **settings)
            actual = HAADFFocusModel.apply_depth_planes_defocus(gpu, **settings)
            np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)

    def test_batched_gpu_projection_matches_tilted_cpu_geometry(self):
        try:
            import cupy as cp
            if not cp.cuda.runtime.getDeviceCount():
                self.skipTest('CUDA unavailable')
        except ImportError:
            self.skipTest('CuPy unavailable')
        sample = self.sample
        sample.set_stage_tilt(Geometry.FloatPoint(x=.15, y=-.2))
        args = sample.initial_view[0], Geometry.FloatSize(10000, 10000), Geometry.FloatPoint(), Geometry.FloatPoint(), Geometry.IntSize(128, 128)
        sample._projection_key = None
        with mock.patch.object(SimulationSettings, 'STL_SURFACE_BACKEND', 'cpu'):
            expected = sample.thickness_projection(*args)
        sample._projection_key = None
        with mock.patch.object(SimulationSettings, 'STL_SURFACE_BACKEND', 'gpu'):
            actual = sample.thickness_projection(*args)
        self.assertIsNotNone(sample._gpu_particle_rasterizer)
        self.assertFalse(sample._gpu_particle_failed)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=.001)

    def test_cropped_ronchigram_matches_independent_full_frame_composition(self):
        from scipy.ndimage import map_coordinates
        fixture = KikuchiModel_test.TestKikuchiModel()
        fixture.sample = self.sample
        manager, camera, context, area = fixture.make_camera(512)
        manager.set_value_2d('stage_position_m', self.sample.initial_view[0])
        manager.set_value('C10Control', 20000e-9)
        try:
            with mock.patch.object(SimulationSettings, 'RONCHIGRAM_BACKEND', 'cpu'):
                frame = camera.get_frame_data(area, Geometry.IntSize(1, 1), .1, context, Geometry.FloatPoint(.5, .5))
                controller = camera._RonchigramCameraSimulator__aberrations_controller
                coordinates = controller.ray_coordinates()
                source = camera._source_image(area, Geometry.IntSize(1, 1))
                base = map_coordinates(source, coordinates, order=1)
                optimized, metadata = camera._apply_multiple_crystals(base, area, Geometry.IntSize(1, 1),
                    camera._last_frame_settings, context, {}, self.sample)
                reference = base.copy()
                for identifier in frame.metadata['kikuchi_simulation']['visible_particle_ids']:
                    crystal = self.sample.crystal(identifier-1)
                    projected = map_coordinates(camera._source_image(area, Geometry.IntSize(1, 1),
                        crystalline=True, sample=crystal), coordinates, order=1)
                    composed, info = camera._apply_kikuchi(base, area, Geometry.IntSize(1, 1),
                        camera._last_frame_settings, context, {}, sample=crystal, crystal_data=projected)
                    reference += composed-base
                np.testing.assert_allclose(optimized, reference.clip(0), rtol=3e-5, atol=.003)
        finally:
            camera.close()

    def test_large_auto_view_reuses_reflectors_and_handles_mixed_backends(self):
        try:
            import cupy as cp
            if not cp.cuda.runtime.getDeviceCount():
                self.skipTest('CUDA unavailable')
        except ImportError:
            self.skipTest('CuPy unavailable')
        fixture = KikuchiModel_test.TestKikuchiModel()
        fixture.sample = self.sample
        manager, camera, context, area = fixture.make_camera(512)
        manager.set_value_2d('stage_position_m', self.sample.initial_view[0])
        manager.set_value('stage_z_m', 100000e-9)
        try:
            with mock.patch.object(SimulationSettings, 'RONCHIGRAM_BACKEND', 'auto'):
                first = camera.get_frame_data(area, Geometry.IntSize(1, 1), .1, context, Geometry.FloatPoint(.5, .5))
                info = first.metadata['kikuchi_simulation']
                self.assertEqual(info['mapping_backend'], 'gpu')
                self.assertLess(info['mapped_particle_count'], len(info['visible_particle_ids']))
                self.assertIn('cpu', {c['compute_backend'] for c in info['crystals']})
                with mock.patch.object(KikuchiModel, 'bands', wraps=KikuchiModel.bands) as reflectors:
                    manager.set_value('stage_z_m', 100001e-9)
                    next_frame = camera.get_frame_data(area, Geometry.IntSize(1, 1), .1, context, Geometry.FloatPoint(.5, .5))
                    self.assertEqual(reflectors.call_count, 0)
                self.assertTrue(np.isfinite(next_frame.data).all())
        finally:
            camera.close()
