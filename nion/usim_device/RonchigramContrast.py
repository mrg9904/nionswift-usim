"""Cached full-resolution Ronchigram composition with optional CUDA execution."""
from __future__ import annotations

import logging
import numpy as np
from scipy.ndimage import gaussian_filter
from nion.usim_device import SimulationSettings, KikuchiModel


class GPUMapper:
    """Keep specimen and nonlinear ray coordinates resident between moves."""

    def __init__(self):
        import cupy as cp
        from cupyx.scipy.ndimage import map_coordinates
        if not cp.cuda.runtime.getDeviceCount():
            raise RuntimeError("No CUDA device available")
        self.cp = cp
        self.map_coordinates = map_coordinates
        self.source = self.coordinates = None
        self.device_source = self.device_coordinates = None

    def apply(self, source, coordinates, dy, dx):
        if source is not self.source:
            self.device_source = self.cp.asarray(source)
            self.source = source
        if coordinates is not self.coordinates:
            self.device_coordinates = self.cp.asarray(coordinates)
            self.coordinates = coordinates
        offsets = self.cp.asarray([dy, dx])[:, None, None]
        return self.map_coordinates(self.device_source, self.device_coordinates + offsets,
                                    order=1, mode="constant", prefilter=False)


class GPULineRenderer:
    """Evaluate all paired Bragg cones in one kernel, reusing it across tilts."""
    def __init__(self):
        import cupy as cp
        if not cp.cuda.runtime.getDeviceCount():
            raise RuntimeError('No CUDA device available')
        self.cp = cp
        self.kernel = cp.RawKernel(r'''
            extern "C" __global__ void lines(const double* bands, int count,
                const double* x, const double* y, int width, int height,
                double line_width, float* output) {
                int pixel = blockIdx.x * blockDim.x + threadIdx.x;
                if (pixel >= width*height) return;
                double a = x[pixel % width], b = y[pixel / width];
                double norm = sqrt(1+a*a+b*b);
                float sum = 0;
                for (int i=0; i<count; ++i) {
                    const double* band = bands+i*5;
                    double distance = (band[0]*a+band[1]*b+band[2])/norm;
                    for (int sign=-1; sign<=1; sign+=2) {
                        double delta = (distance-sign*band[3])/line_width;
                        if (fabs(delta)<4) sum = (float)((double)sum+sign*band[4]*exp(-.5*delta*delta));
                    }
                }
                output[pixel] = tanhf(sum);
            }
        ''', 'lines', options=('--fmad=false',))
        self.resize_kernel = cp.RawKernel(r'''
            extern "C" __global__ void resize_lines(const float* input,
                int old_width, int old_height, int width, int height, float* output) {
                int pixel = blockIdx.x*blockDim.x+threadIdx.x;
                if (pixel>=width*height) return;
                double x = (double)(pixel%width)*(old_width-1)/max(width-1, 1);
                double y = (double)(pixel/width)*(old_height-1)/max(height-1, 1);
                int left = (int)x, top = (int)y;
                int right = min(left+1, old_width-1), bottom = min(top+1, old_height-1);
                double a = x-left, b = y-top;
                output[pixel] = (float)((1-b)*((1-a)*input[top*old_width+left]+a*input[top*old_width+right])
                    +b*((1-a)*input[bottom*old_width+left]+a*input[bottom*old_width+right]));
            }
        ''', 'resize_lines', options=('--fmad=false',))

    def render(self, bands, x_rad, y_rad, line_width=.00035, output_shape=None):
        cp = self.cp
        parameters = np.asarray([[*band.normal, np.sin(band.theta_b_rad), band.weight] for band in bands], dtype=np.float64)
        result = cp.empty((len(y_rad), len(x_rad)), cp.float32)
        self.kernel(((result.size+255)//256,), (256,), (cp.asarray(parameters), np.int32(len(bands)),
            cp.asarray(np.tan(x_rad)), cp.asarray(np.tan(y_rad)), np.int32(len(x_rad)), np.int32(len(y_rad)), np.float64(line_width), result))
        if output_shape is not None and tuple(output_shape) != result.shape:
            resized = cp.empty(output_shape, cp.float32)
            self.resize_kernel(((resized.size+255)//256,), (256,), (result,
                np.int32(result.shape[1]), np.int32(result.shape[0]), np.int32(output_shape[1]), np.int32(output_shape[0]), resized))
            result = resized
        return cp.asnumpy(result)


class ContrastComposer:
    """Own one pattern's blur bank; moving the probe does not rebuild it.

    GPU imports are lazy. A failed CUDA operation releases this camera's device
    references and retries on CPU, without changing the simulation geometry.
    """

    def __init__(self, pattern, pixel_angles_rad, backend="auto"):
        self.pattern = np.asarray(pattern, dtype=np.float32)
        self.pixel_angles = np.abs(np.asarray(pixel_angles_rad, dtype=float))
        if np.any(self.pixel_angles <= 0):
            raise ValueError("Positive pixel angles required")
        if backend not in ("auto", "cpu", "gpu"):
            raise ValueError("Ronchigram backend must be auto, cpu or gpu")
        self.backend = "cpu"
        self.fallback_reason = None
        self.xp = np
        self.filter = gaussian_filter
        self.levels = [0., 10., 25., 50., 100.]
        self.bank = None
        # Bound persistent storage; no global cache accumulates old tilts.
        budget = SimulationSettings.RONCHIGRAM_PATTERN_CACHE_BYTES
        self.max_levels = max(5, budget // self.pattern.nbytes)
        if backend == "gpu" or (backend == "auto" and self.pattern.size >= SimulationSettings.RONCHIGRAM_GPU_MINIMUM_PIXELS):
            try:
                import cupy as cp
                from cupyx.scipy.ndimage import gaussian_filter as gpu_filter
                if not cp.cuda.runtime.getDeviceCount():
                    raise RuntimeError("No CUDA device available")
                self.xp, self.filter, self.backend = cp, gpu_filter, "gpu"
                # Build once at the actual maximum thickness during compose;
                # a fixed 100 nm prebuild was repeated for thicker cathodes.
            except Exception as error:
                self._fallback(error)

    def _fallback(self, error):
        self.fallback_reason = f"{type(error).__name__}: {error}"
        logging.getLogger(__name__).warning("Ronchigram CUDA unavailable; using CPU: %s", self.fallback_reason)
        self.xp, self.filter, self.backend = np, gaussian_filter, "cpu"
        self.bank = None

    def _prepare(self, maximum):
        levels = [0., 10., 25., 50., 100.]
        while levels[-1] < maximum and len(levels) < self.max_levels:
            levels.append(2*levels[-1])
        if self.bank is not None and levels == self.levels:
            return
        self.levels = levels
        xp = self.xp
        pattern = xp.asarray(self.pattern)
        images = [pattern]
        for level in levels[1:]:
            sigma = SimulationSettings.KIKUCHI_BROADENING_RAD_AT_100_NM * np.sqrt(level/100.) / self.pixel_angles
            images.append(self.filter(pattern, sigma, mode="nearest"))
        self.bank = xp.stack(images)

    def compose(self, real_image, vacuum_level):
        if vacuum_level <= 0:
            raise ValueError("Positive vacuum intensity required")
        try:
            return self._compose(real_image, vacuum_level)
        except Exception as error:
            if self.backend != "gpu":
                raise
            self._fallback(error)
            return self._compose(real_image, vacuum_level)

    def _compose(self, real_image, vacuum_level):
        xp = self.xp
        if xp is np and not isinstance(real_image, np.ndarray):
            real_image = real_image.get()
        real = xp.clip(xp.asarray(real_image, dtype=xp.float32), 0, vacuum_level)
        thickness = -SimulationSettings.RONCHIGRAM_TRANSMISSION_LENGTH_NM * xp.log(xp.clip(real/vacuum_level, 1e-12, 1.))
        thickness = xp.where(real > 0, thickness, 0.)
        maximum = float(thickness.max())
        self._prepare(maximum)
        diffuse = -xp.expm1(-thickness/SimulationSettings.RONCHIGRAM_DIFFUSE_LENGTH_NM)
        halo = self.filter(real, SimulationSettings.RONCHIGRAM_DIFFUSE_SIGMA_RAD/self.pixel_angles, mode="nearest")
        base = (1-diffuse)*real + diffuse*(.7*halo + .3*vacuum_level)
        levels = xp.asarray(self.levels, dtype=xp.float32)
        lower = xp.clip(xp.searchsorted(levels, thickness, side="right")-1, 0, len(self.levels)-2)
        upper = lower+1
        fraction = xp.clip((thickness-levels[lower])/(levels[upper]-levels[lower]), 0, 1)
        # Gather just two neighbours per pixel, rather than a full-frame weight
        # array and multiply for every thickness level.
        pixels = xp.arange(real.size).reshape(real.shape)
        bank = self.bank.reshape(len(self.levels), -1)
        blurred = (1-fraction)*bank[lower, pixels] + fraction*bank[upper, pixels]
        amplitude = KikuchiModel.thickness_contrast(thickness, xp=xp)
        result = xp.maximum(base + vacuum_level*amplitude*blurred, 0)
        metadata = {"projected_thickness_range_nm": [float(thickness.min()), maximum],
                    "diffuse_fraction_range": [float(diffuse.min()), float(diffuse.max())],
                    "compute_backend": self.backend}
        if self.fallback_reason:
            metadata["gpu_fallback_reason"] = self.fallback_reason
        return (xp.asnumpy(result) if self.backend == "gpu" else result), metadata
