# standard libraries
import numpy
import numpy.typing
import typing
import math

from nion.data import DataAndMetadata
from nion.usim_device import SimulationSettings

_NDArray = numpy.typing.NDArray[typing.Any]


class HAADFNoise:
    """Electron-counting shot noise with optional additive detector read noise.

    Return integrated signal in the historical uSim intensity * microsecond
    units, referenced to 200 pA. Its mean grows with dose; its relative noise
    shrinks as 1/sqrt(dwell). A normalized image is obtained by dividing by
    dwell, rather than reducing the shot fluctuations in raw counts.
    """

    def __init__(self, seed: typing.Optional[int] = None) -> None:
        self._rng = numpy.random.default_rng(seed)
        self._seed = seed
        self._gpu_rng: typing.Any = None

    def apply(self, image: typing.Any, pixel_time_us: float, beam_current_a: float) -> typing.Any:
        if not math.isfinite(pixel_time_us) or pixel_time_us <= 0:
            raise ValueError("pixel_time_us must be finite and positive")
        if not math.isfinite(beam_current_a) or beam_current_a < 0:
            raise ValueError("beam_current_a must be finite and non-negative")
        efficiency = SimulationSettings.HAADF_DETECTION_EFFICIENCY
        if not math.isfinite(efficiency) or not 0 < efficiency <= 1:
            raise ValueError("HAADF_DETECTION_EFFICIENCY must be in (0, 1]")
        read_noise = SimulationSettings.HAADF_READ_NOISE_ELECTRONS
        if not math.isfinite(read_noise) or read_noise < 0:
            raise ValueError("HAADF_READ_NOISE_ELECTRONS must be finite and non-negative")
        xp: typing.Any = numpy
        rng: typing.Any = self._rng
        if hasattr(image, "__cuda_array_interface__"):
            import cupy
            xp = cupy
            if self._gpu_rng is None:
                self._gpu_rng = cupy.random.RandomState(self._seed)
            rng = self._gpu_rng
        # 200 pA is the existing simulator's reference beam current.
        reference_rate = 200e-12 / 1.602176634e-19 * 1e-6 * efficiency
        dose = beam_current_a / 200e-12 * pixel_time_us
        ideal = xp.maximum(xp.asarray(image, dtype=xp.float32), 0)
        if not SimulationSettings.HAADF_SHOT_NOISE_ENABLED and read_noise == 0:
            return (ideal * dose).astype(xp.float32)
        electrons = ideal * (reference_rate * dose)
        if SimulationSettings.HAADF_SHOT_NOISE_ENABLED:
            electrons = rng.poisson(electrons).astype(xp.float32)
        if read_noise > 0:
            electrons += rng.normal(0.0, read_noise, size=image.shape).astype(xp.float32)
        return (electrons / reference_rate).astype(xp.float32)


class PoissonNoise:

    def __init__(self) -> None:
        self.enabled = True
        self.poisson_level: typing.Optional[float] = None

    def apply(self, input: DataAndMetadata.DataAndMetadata, lambda_thresh: float = 1.0) -> DataAndMetadata.DataAndMetadata:
        if self.enabled and self.poisson_level:
            rs = numpy.random.RandomState()  # use this to avoid blocking other calls to poisson
            input_data = input.data
            input_data_shape = input.data_shape
            assert input_data is not None
            if self.poisson_level > lambda_thresh:
                # Since it is 'high' lambda, we can approximate it to a normal distribution
                poisson_data = typing.cast(_NDArray, rs.normal(loc=self.poisson_level, scale=numpy.sqrt(self.poisson_level), size=input_data_shape).astype(input_data.dtype))
            else:
                poisson_data = typing.cast(_NDArray, rs.poisson(self.poisson_level, size=input_data_shape).astype(input_data.dtype))
            return input + (poisson_data - self.poisson_level)
        return input

