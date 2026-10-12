"""Batch SI is consistent with the live spectrum and preserves missing losses."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
from nion.utils import Geometry
from nion.swift.test import TestContext
from nion.instrumentation import stem_controller
from nion.instrumentation.test import AcquisitionTestContext
from nion.usim_device import DeviceConfiguration, EELSModel, EELSSpectrumImaging


class TestEELSSpectrumImaging(unittest.TestCase):
    def test_batch_matches_live_camera_and_cropped_energy_window(self):
        setup = TestContext.TestSetup()
        configuration = DeviceConfiguration.AcquisitionContextConfiguration(sample_index=8, set_configuration_location=False)
        with tempfile.TemporaryDirectory() as directory:
            configuration.configuration_location = directory
            with AcquisitionTestContext.AcquisitionTestContext(configuration) as context:
                instrument = context.instrument
                simulator = instrument.value_manager.eels_camera.simulator
                simulator.noise.enabled = False
                instrument.value_manager.energy_offset_eV = -10.
                instrument.value_manager.energy_per_channel_eV = 2.
                n, fov = 16, 2000.
                parameters = dict(pixel_size=(n, n), pixel_time_us=10., fov_nm=fov, rotation_rad=0., center_nm=(0., 0.))
                output, energies, metadata = simulator.acquire_spectrum_image(parameters,
                    output_path=Path(directory)/'cube.npy', energy_channel_count=255)
                self.assertEqual(output.shape, (16, 16, 255))
                self.assertEqual(energies[-1], 498.)
                scan = stem_controller.ScanContext(Geometry.IntSize(n, n), Geometry.FloatPoint(), fov, 0.)
                area = Geometry.IntRect(origin=Geometry.IntPoint(), size=Geometry.IntSize(256, 1024))
                for row, column in ((4, 4), (8, 8), (11, 9)):
                    position = Geometry.FloatPoint((row+.5)/n, (column+.5)/n)
                    instrument.probe_position = position
                    live = simulator.get_frame_data(area, Geometry.IntSize(1, 1), 10e-6, scan, position).data.sum(axis=0)
                    # Optical-depth binning is bounded to .005 per material.
                    self.assertLess(float(np.abs(output[row, column]-live[:255]).sum()/simulator.get_total_counts(10e-6)), .025)
                    self.assertLessEqual(float(output[row, column].sum()), simulator.get_total_counts(10e-6)*1.001)
                self.assertEqual(metadata['status'], 'complete')
                self.assertEqual(metadata['pixel_time_us'], 10.)
                files_before = {path.name for path in Path(directory).iterdir()}
                transient, _, transient_metadata = simulator.acquire_spectrum_image(parameters,
                    output_path=None, energy_channel_count=255)
                np.testing.assert_array_equal(transient, output)
                self.assertNotIsInstance(transient, np.memmap)
                self.assertFalse(transient_metadata['spectrum_image_saved'])
                self.assertEqual({path.name for path in Path(directory).iterdir()}, files_before)
                self.assertTrue((Path(directory)/'cube.npy').exists())
                with mock.patch.object(EELSModel, 'spectrum_probabilities', wraps=EELSModel.spectrum_probabilities) as model:
                    doubled, _, _ = simulator.acquire_spectrum_image(dict(parameters, pixel_time_us=20.),
                        output_path=Path(directory)/'warm.npy', energy_channel_count=255)
                    self.assertEqual(model.call_count, 0)
                    np.testing.assert_allclose(doubled, 2*output, rtol=2e-6)
                    del doubled
                del output

    def test_log_ratio_wide_and_short_windows_against_forward_model(self):
        energy = -10.+np.arange(1024)*4.
        short = -10.+np.arange(255)*2.
        for tau in (.25, .75, 1.5, 3.):
            layer = EELSModel.EELSLayer(100*tau, EELSModel.EELSMaterial())
            wide, _ = EELSModel.spectrum_probabilities([layer], energy, 4.)
            cropped, _ = EELSModel.spectrum_probabilities([layer], short, 2.)
            ratio = np.log(wide.sum()/wide[abs(energy) <= 5].sum())
            narrow_ratio = np.log(cropped.sum()/cropped[abs(short) <= 5].sum())
            self.assertAlmostEqual(ratio, tau, delta=.03)
            self.assertLess(narrow_ratio, ratio)
