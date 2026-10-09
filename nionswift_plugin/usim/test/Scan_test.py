import contextlib
import unittest
from nion.swift import Facade
from nion.utils import Geometry
from nion.instrumentation.test import ScanControl_test
from nion.instrumentation.test import AcquisitionTestContext
from nion.usim_device import DeviceConfiguration


class TestSimulatorScan(ScanControl_test.TestScanControlClass):

    def _test_context(self, *, is_eels: bool = False) -> AcquisitionTestContext.AcquisitionTestContext:
        context = AcquisitionTestContext.AcquisitionTestContext(DeviceConfiguration.AcquisitionContextConfiguration(), is_eels=is_eels)
        # Parent instrumentation tests require 100 nm; uSim's non-STL default is 200 nm.
        for index in range(3):
            parameters = context.scan_hardware_source.get_frame_parameters(index)
            parameters.fov_nm = 100.
            context.scan_hardware_source.set_frame_parameters(index, parameters)
        context.scan_hardware_source.set_current_frame_parameters(context.scan_hardware_source.get_frame_parameters(0))
        return context

    def test_selecting_samples_restores_their_own_views_and_profiles(self) -> None:
        with self._test_context() as context:
            source = context.scan_hardware_source
            instrument = context.instrument
            generator = instrument.scan_data_generator
            for index in (6, 5, 0, 1, 2, 3, 4, 6):
                generator.sample_index = index
                expected_stage = Geometry.FloatPoint(y=279e-9, x=1222e-9) if index == 6 else Geometry.FloatPoint()
                expected_fov = 10000. if index == 6 else 200.
                self.assertEqual(instrument.stage_position_m, expected_stage)
                self.assertEqual(source.get_current_frame_parameters().fov_nm, expected_fov)
                self.assertEqual(source.get_record_frame_parameters().fov_nm, expected_fov)
                for profile in range(3):
                    self.assertEqual(source.get_frame_parameters(profile).fov_nm, expected_fov)
                    self.assertEqual(source.get_frame_parameters(profile).center_nm, Geometry.FloatPoint())
                # Selecting the active specimen again must not reset a user's view.
                parameters = source.get_current_frame_parameters()
                parameters.fov_nm = 75.
                source.set_current_frame_parameters(parameters)
                generator.sample_index = index
                self.assertEqual(source.get_current_frame_parameters().fov_nm, 75.)

    def test_facade_record_data_with_immediate_close(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            api = Facade.get_api("~1.0", "~1.0")
            hardware_source_facade = api.get_hardware_source_by_id(scan_hardware_source.hardware_source_id, "~1.0")
            assert hardware_source_facade
            scan_frame_parameters = hardware_source_facade.get_frame_parameters_for_profile_by_index(2)
            scan_frame_parameters["external_clock_wait_time_ms"] = 20000 # int(camera_frame_parameters["exposure_ms"] * 1.5)
            scan_frame_parameters["external_clock_mode"] = 1
            scan_frame_parameters["ac_line_sync"] = False
            scan_frame_parameters["ac_frame_sync"] = False
            # this tests an issue for a race condition where thread for record task isn't started before the task
            # is canceled, resulting in the close waiting for the thread and the thread waiting for the acquire.
            # this reduces the problem, but it's still possible that during external sync, the acquisition starts
            # before being canceled and must timeout.
            with contextlib.closing(hardware_source_facade.create_record_task(scan_frame_parameters)) as task:
                pass


if __name__ == '__main__':
    unittest.main()
