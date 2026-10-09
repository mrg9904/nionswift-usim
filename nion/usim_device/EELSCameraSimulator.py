from __future__ import annotations

# standard libraries
import math
import typing

import numpy
import numpy.typing
import scipy.ndimage.interpolation
import scipy.stats

from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.device_kit import InstrumentDevice
from nion.instrumentation import stem_controller
from nion.usim_device import EELSModel
from nion.usim_device import CameraSimulator
from nion.usim_device import InstrumentDevice as InstrumentDevice_
from nion.usim_device import SampleSimulator
from nion.usim_device import Noise
from nion.utils import Geometry


_NDArray = numpy.typing.NDArray[typing.Any]


def plot_powerlaw(data: _NDArray, multiplier: float, energy_calibration: Calibration.Calibration, offset_ev: float, onset_ev: float) -> None:
    # calculate the range
    # 1 represents 0eV, 0 represents 4000eV
    # TODO: sub-pixel accuracy
    energy_range_ev = [energy_calibration.convert_to_calibrated_value(0),
                       energy_calibration.convert_to_calibrated_value(data.shape[0])]
    envelope = scipy.stats.norm(loc=offset_ev, scale=onset_ev).cdf(numpy.linspace(energy_range_ev[0], energy_range_ev[1], data.shape[0]))
    max_ev = 4000
    powerlaw_dist = scipy.stats.powerlaw(8, loc=0, scale=max_ev)  # this is an increasing function; must be reversed below; 8 is arbitrary but looks good
    powerlaw = powerlaw_dist.pdf(numpy.linspace(max_ev - energy_range_ev[0], max_ev - energy_range_ev[1], data.shape[0]))
    data += envelope * multiplier * powerlaw


def plot_norm(data: _NDArray, multiplier: float, energy_calibration: Calibration.Calibration, energy_ev: float, energy_width_ev: float) -> None:
    # calculate the range
    # 1 represents 0eV, 0 represents 4000eV
    # TODO: sub-pixel accuracy
    data_range = [0, data.shape[0]]
    energy_range_ev = [energy_calibration.convert_to_calibrated_value(data_range[0]),
                       energy_calibration.convert_to_calibrated_value(data_range[1])]
    norm = scipy.stats.norm(loc=energy_ev, scale=energy_width_ev)
    data += multiplier * norm.pdf(numpy.linspace(energy_range_ev[0], energy_range_ev[1], data_range[1])) / norm.pdf(energy_ev)


def plot_spectrum(feature: SampleSimulator.Feature, data: _NDArray, multiplier: float, energy_calibration: Calibration.Calibration) -> None:
    for edge_eV, onset_eV in feature.edges:
        # print(f"edge_eV {edge_eV} onset_eV {onset_eV}")
        strength = multiplier * 0.1
        plot_powerlaw(data, strength, energy_calibration, edge_eV, onset_eV)
    for n in range(1, feature.plurality + 1):
        plot_norm(data, multiplier / math.factorial(n), energy_calibration, feature.plasmon_eV * n, math.sqrt(feature.plasmon_eV))


class EELSCameraSimulator(CameraSimulator.CameraSimulator):
    depends_on = ["is_slit_in", "probe_state", "probe_position", "is_blanked", "ZLPoffset",
                  "stage_position_m", "beam_shift_m", "features", "energy_offset_eV", "energy_per_channel_eV",
                  "BeamCurrent"]

    def __init__(self, instrument: InstrumentDevice_.Instrument, sensor_dimensions: Geometry.IntSize, counts_per_electron: int) -> None:
        super().__init__(instrument, "eels", sensor_dimensions, counts_per_electron)
        self.__cached_frame: typing.Optional[DataAndMetadata.DataAndMetadata] = None
        self.noise = Noise.EELSShotNoise(counts_per_electron)

    def get_frame_data(self, readout_area: Geometry.IntRect, binning_shape: Geometry.IntSize, exposure_s: float, scan_context: stem_controller.ScanContext, parked_probe_position: typing.Optional[Geometry.FloatPoint]) -> DataAndMetadata.DataAndMetadata:
        """Generate geometry-driven EELS with thickness-dependent plural scattering."""

        frame_settings = self._get_frame_settings(readout_area, binning_shape, exposure_s, scan_context, parked_probe_position)
        if frame_settings != self._last_frame_settings:
            self._needs_recalculation = True
            self._last_frame_settings = frame_settings

        if self._needs_recalculation or self.__cached_frame is None:
            data: numpy.typing.NDArray[numpy.float64] = numpy.zeros(tuple(self._sensor_dimensions), float)
            value_manager = typing.cast("InstrumentDevice_.ValueManager", self.instrument.value_manager)
            slit_attenuation = 10 if value_manager.is_slit_in else 1
            intensity_calibration = Calibration.Calibration(units="counts")
            dimensional_calibrations = self.get_dimensional_calibrations(readout_area, Geometry.IntSize(1, 1))

            target_pixel_count = self.get_total_counts(exposure_s) / data.shape[0]
            metadata = {}
            if frame_settings.current_probe_position is not None:
                position = EELSModel.probe_sample_position(
                    value_manager.actual_offset_m,
                    scan_context.fov_size_nm or Geometry.FloatSize(),
                    scan_context.center_nm or Geometry.FloatPoint(),
                    frame_settings.current_probe_position, scan_context.rotation_rad)
                generator = typing.cast(InstrumentDevice_.ScanDataGenerator, self.instrument.scan_data_generator)
                layers = generator.sample.eels_layers_at(position)
                calibration = dimensional_calibrations[1]
                energies = calibration.offset + numpy.arange(data.shape[1]) * calibration.scale
                probabilities, metadata = EELSModel.spectrum_probabilities(
                    layers, energies, calibration.scale, .5/slit_attenuation)
                data[:] = probabilities * target_pixel_count / slit_attenuation
                metadata["probe_position_sample_m"] = {"x": position.x, "y": position.y}
            data = self._get_binned_data(data, binning_shape)

            self.__cached_frame = DataAndMetadata.new_data_and_metadata(data.astype(numpy.float32), intensity_calibration=intensity_calibration, dimensional_calibrations=self.get_dimensional_calibrations(readout_area, binning_shape), metadata={"eels_simulation": metadata})
            self._needs_recalculation = False

        return self.noise.apply(self.__cached_frame)

    def _get_binned_data(self, data, binning_shape):
        """Sum energy and detector rows independently, preserving counts."""
        rows, columns = binning_shape.height, binning_shape.width
        height, width = data.shape[0]//rows, data.shape[1]//columns
        return data[:height*rows, :width*columns].reshape(height, rows, width, columns).sum(axis=(1, 3))

    def get_dimensional_calibrations(self, readout_area: typing.Optional[Geometry.IntRect], binning_shape: typing.Optional[Geometry.IntSize]) -> typing.Sequence[Calibration.Calibration]:
        value_manager = typing.cast("InstrumentDevice_.ValueManager", self.instrument.value_manager)
        energy_offset_ev = value_manager.energy_offset_eV
        # energy_offset_ev += random.uniform(-1, 1) * self.__energy_per_channel_eV * 5
        dimensional_calibrations = [
            Calibration.Calibration(),
            Calibration.Calibration(offset=energy_offset_ev + ((binning_shape.width if binning_shape else 1)-1)*value_manager.energy_per_channel_eV/2,
                scale=value_manager.energy_per_channel_eV*(binning_shape.width if binning_shape else 1), units="eV")
        ]
        return dimensional_calibrations
