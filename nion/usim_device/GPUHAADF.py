"""Device-resident STL slicing and reflected-boundary HAADF filtering."""

from __future__ import annotations

import math
import typing

import numpy
import numpy.typing

from nion.usim_device import SimulationSettings


class GPUPreparedDepthPlanes(typing.Sequence[typing.Tuple[float, numpy.typing.NDArray[numpy.float32]]]):
    """Generate slices on demand, retaining spectra within a GPU memory budget.

    No full depth stack is transferred to host memory. The CPU Sequence
    interface is retained for diagnostics; the acquisition path uses render.
    """

    _on_gpu = True

    def __init__(self, lower: typing.Any, upper: typing.Any, slice_thickness_nm: float) -> None:
        import cupy as cp
        from cupyx.scipy import fft

        self._cp: typing.Any = cp
        self._fft: typing.Any = fft
        # Runtime radius avoids compiling a new ndimage kernel for every
        # near-focus sigma encountered while changing FoV.
        self._gaussian = cp.RawKernel(r'''
            extern "C" __global__ void gaussian(
                const float* input, float* output, const float* weights,
                int radius, int height, int width, int axis) {
                int pixel = blockIdx.x * blockDim.x + threadIdx.x;
                if (pixel >= height * width) return;
                int row = pixel / width, col = pixel % width;
                int length = axis == 0 ? height : width;
                int center = axis == 0 ? row : col;
                float sum = 0.0f;
                for (int k = -radius; k <= radius; ++k) {
                    int position = (center + k) % (2 * length);
                    if (position < 0) position += 2 * length;
                    if (position >= length) position = 2 * length - 1 - position;
                    int source = axis == 0 ? position * width + col : row * width + position;
                    sum += input[source] * weights[k + radius];
                }
                output[pixel] = sum;
            }
        ''', "gaussian", options=("--fmad=false",))
        self._lower = cp.asarray(lower, dtype=cp.float32)
        self._upper = cp.asarray(upper, dtype=cp.float32)
        self.shape = tuple(self._lower.shape)
        self._slice = slice_thickness_nm
        self._normalize = SimulationSettings.STL_NORMALIZE_COLUMN_INTENSITY
        self._reference = SimulationSettings.STL_REFERENCE_THICKNESS_NM
        self._column_intensity = SimulationSettings.STL_NORMALIZED_COLUMN_INTENSITY
        valid = cp.isfinite(self._lower) & cp.isfinite(self._upper) & (self._upper > self._lower)
        self._empty = not bool(cp.any(valid))
        if self._empty:
            self._first = 0
            self._count = 1
        else:
            self._first = math.floor(float(cp.min(cp.where(valid, self._lower, cp.inf))) / self._slice)
            final = math.ceil(float(cp.max(cp.where(valid, self._upper, -cp.inf))) / self._slice)
            self._count = max(1, final - self._first)
        self._spectra: typing.Dict[int, typing.Any] = {}
        self._cached_bytes = 0
        self._frequency_y = (cp.pi * cp.arange(self.shape[0], dtype=cp.float64) / self.shape[0]) ** 2
        self._frequency_x = (cp.pi * cp.arange(self.shape[1], dtype=cp.float64) / self.shape[1]) ** 2
        self._slice_kernel = cp.ElementwiseKernel(
            "float32 lower, float32 upper, float64 zlow, float64 dz, "
            "float32 reference, bool normalize, float32 column_intensity",
            "float32 plane",
            """
            if (!isfinite(lower) || !isfinite(upper) || upper <= lower) {
                plane = 0.0f;
            } else {
                float overlap = (float)fmax(0.0, fmin((double)upper, zlow+dz)
                                                  - fmax((double)lower, zlow));
                plane = normalize ? overlap / (upper-lower) * column_intensity
                                  : overlap / reference;
            }
            """,
            "usim_depth_slice",
        )

    def __len__(self) -> int:
        return self._count

    def depth(self, index: int) -> float:
        return 0.0 if self._empty else (self._first + index + 0.5) * self._slice

    @typing.overload
    def __getitem__(self, index: int) -> typing.Tuple[float, numpy.typing.NDArray[numpy.float32]]: ...

    @typing.overload
    def __getitem__(self, index: slice) -> typing.Sequence[typing.Tuple[float, numpy.typing.NDArray[numpy.float32]]]: ...

    def __getitem__(self, index: typing.Union[int, slice]) -> typing.Any:
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        plane = self._cp.asnumpy(self.plane(index))
        plane.setflags(write=False)
        return self.depth(index), plane

    def plane(self, index: int) -> typing.Any:
        return self._slice_kernel(
            self._lower, self._upper, (self._first + index) * self._slice,
            self._slice, self._reference, self._normalize, self._column_intensity,
        )

    def spectrum(self, index: int) -> typing.Any:
        cached = self._spectra.get(index)
        if cached is not None:
            return cached
        spectrum = self._fft.dctn(self.plane(index), norm="ortho")
        if self._cached_bytes + spectrum.nbytes <= SimulationSettings.GPU_DEPTH_SPECTRUM_CACHE_BYTES:
            self._spectra[index] = spectrum
            self._cached_bytes += spectrum.nbytes
        return spectrum

    def blur(self, plane: typing.Any, sigma_y: float, sigma_x: float) -> typing.Any:
        cp = self._cp
        for axis, sigma in enumerate((sigma_y, sigma_x)):
            radius = int(SimulationSettings.GAUSSIAN_TRUNCATE * sigma + .5)
            if not radius:
                continue
            offsets = numpy.arange(-radius, radius + 1, dtype=numpy.float64)
            weights = numpy.exp(-.5 * (offsets / sigma) ** 2)
            weights /= weights.sum()
            output = cp.empty_like(plane)
            self._gaussian(((plane.size + 255) // 256,), (256,),
                           (plane, output, cp.asarray(weights, dtype=cp.float32),
                            numpy.int32(radius), numpy.int32(self.shape[0]),
                            numpy.int32(self.shape[1]), numpy.int32(axis)))
            plane = output
        return plane

    def render(
        self, *, defocus_m: float, best_focus_m: float, convergence_angle_rad: float,
        pixel_size_y_nm: float, pixel_size_x_nm: float, return_device: bool = False,
    ) -> typing.Any:
        from nion.usim_device import HAADFFocusModel

        cp = self._cp
        output = cp.zeros(self.shape, dtype=cp.float32)
        combined = cp.zeros_like(output)
        has_spectrum = False
        for index in range(len(self)):
            sigma_y, sigma_x = HAADFFocusModel._probe_sigma(
                (defocus_m - best_focus_m - self.depth(index) * 1e-9) * 1e9,
                convergence_angle_rad, pixel_size_y_nm, pixel_size_x_nm,
                SimulationSettings.MINIMUM_SIGMA_PX,
            )
            if max(sigma_y, sigma_x) < SimulationSettings.FOURIER_BLUR_THRESHOLD_PX:
                output += self.blur(self.plane(index), sigma_y, sigma_x)
                continue
            rows, columns = HAADFFocusModel._spectral_shape(self.shape, sigma_y, sigma_x)
            weight_y = cp.exp(-0.5 * self._frequency_y[:rows] * sigma_y**2).astype(cp.float32)
            weight_x = cp.exp(-0.5 * self._frequency_x[:columns] * sigma_x**2).astype(cp.float32)
            combined[:rows, :columns] += self.spectrum(index)[:rows, :columns] * weight_y[:, None] * weight_x[None, :]
            has_spectrum = True
        if has_spectrum:
            output += self._fft.idctn(combined, norm="ortho")
        return output if return_device else cp.asnumpy(output)


class GPUPreparedLayerPlanes(GPUPreparedDepthPlanes):
    """One material interval with explicitly bounded, unnormalized depth slices."""
    def __init__(self, lower, upper, slice_thickness_nm, origin_nm, end_nm, *, normalize=False):
        super().__init__(lower, upper, slice_thickness_nm)
        self._normalize = normalize
        self._reference = 20.
        self._column_intensity = 1.
        self._origin = origin_nm
        self._end = end_nm
        self._first = 0
        self._count = max(1, len(numpy.arange(origin_nm, end_nm, slice_thickness_nm)))
        self._batch_spectra = {}

    def depth(self, index):
        low = self._origin+index*self._slice
        return (low+min(low+self._slice, self._end))*.5

    def plane(self, index):
        low = self._origin+index*self._slice
        return self._slice_kernel(self._lower, self._upper, low, min(self._slice, self._end-low),
            self._reference, self._normalize, self._column_intensity)

    def render(self, *, defocus_m, best_focus_m, convergence_angle_rad,
               pixel_size_y_nm, pixel_size_x_nm, return_device=False):
        from nion.usim_device import HAADFFocusModel
        cp = self._cp
        output = cp.zeros(self.shape, cp.float32)
        combined = cp.zeros_like(output)
        sigmas = numpy.asarray([HAADFFocusModel._probe_sigma(
            (defocus_m-best_focus_m-self.depth(i)*1e-9)*1e9, convergence_angle_rad,
            pixel_size_y_nm, pixel_size_x_nm, SimulationSettings.MINIMUM_SIGMA_PX) for i in range(len(self))])
        large = numpy.max(sigmas, axis=1) >= SimulationSettings.FOURIER_BLUR_THRESHOLD_PX
        for i in numpy.flatnonzero(~large):
            output += self.blur(self.plane(int(i)), *sigmas[i])
        # Batch slices and DCTs instead of ~64 independent transforms/launches.
        # Bound temporary stacks to 128 MiB even for a 2048-square record.
        batch_size = max(1, (128*1024*1024)//(self._lower.size*4))
        selected = numpy.flatnonzero(large)
        for start in range(0, len(selected), batch_size):
            indexes = selected[start:start+batch_size]
            key = tuple(indexes)
            spectra = self._batch_spectra.get(key)
            if spectra is None:
                lows = self._origin+indexes*self._slice
                steps = numpy.minimum(self._slice, self._end-lows)
                stack = self._slice_kernel(self._lower[None], self._upper[None],
                    cp.asarray(lows)[:, None, None], cp.asarray(steps)[:, None, None],
                    self._reference, self._normalize, self._column_intensity)
                spectra = self._fft.dctn(stack, axes=(-2, -1), norm='ortho')
                if self._cached_bytes+spectra.nbytes <= SimulationSettings.GPU_DEPTH_SPECTRUM_CACHE_BYTES:
                    self._batch_spectra[key] = spectra
                    self._cached_bytes += spectra.nbytes
            shapes = [HAADFFocusModel._spectral_shape(self.shape, *sigmas[i]) for i in indexes]
            rows, columns = numpy.max(shapes, axis=0)
            ys = cp.asarray(sigmas[indexes, 0])[:, None]
            xs = cp.asarray(sigmas[indexes, 1])[:, None]
            wy = cp.exp(-.5*self._frequency_y[None, :rows]*ys**2).astype(cp.float32)
            wx = cp.exp(-.5*self._frequency_x[None, :columns]*xs**2).astype(cp.float32)
            limits = cp.asarray(shapes)
            wy *= cp.arange(rows)[None, :] < limits[:, 0, None]
            wx *= cp.arange(columns)[None, :] < limits[:, 1, None]
            combined[:rows, :columns] += cp.sum(spectra[:, :rows, :columns]*wy[:, :, None]*wx[:, None, :], axis=0)
        if len(selected):
            output += self._fft.idctn(combined, norm='ortho')
        return output if return_device else cp.asnumpy(output)


class GPUSparseDepthPlanes(GPUPreparedDepthPlanes):
    """Accumulate sparse independent intervals on GPU, preserving air gaps."""
    def __init__(self, shape, intervals, depth_limits, copper):
        empty = numpy.full(shape, numpy.nan, numpy.float32)
        super().__init__(empty, empty, 1.)
        self._depth_limits = depth_limits
        self._copper_depth, copper_image = copper
        self._copper = self._cp.asarray(copper_image)
        self._count = 1+len(depth_limits)
        pixels, lows, highs = [], [], []
        for ys, xs, lower, upper in intervals:
            valid = numpy.isfinite(lower) & numpy.isfinite(upper) & (upper > lower)
            y, x = numpy.nonzero(valid)
            pixels.append((y+(ys.start or 0))*shape[1]+x+(xs.start or 0))
            lows.append(lower[valid])
            highs.append(upper[valid])
        self._pixels = self._cp.asarray(numpy.concatenate(pixels) if pixels else [], dtype=self._cp.int32)
        self._lows = self._cp.asarray(numpy.concatenate(lows) if lows else [], dtype=self._cp.float32)
        self._highs = self._cp.asarray(numpy.concatenate(highs) if highs else [], dtype=self._cp.float32)
        self._accumulate = self._cp.RawKernel(r'''
            extern "C" __global__ void intervals(const int* pixels,
                const float* lo, const float* hi, int count, double bottom,
                double top, float* plane) {
                int i=blockIdx.x*blockDim.x+threadIdx.x;
                if (i>=count) return;
                float overlap=(float)fmax(0., fmin((double)hi[i],top)-fmax((double)lo[i],bottom));
                if (overlap>0) atomicAdd(plane+pixels[i], overlap/20.f);
            }
        ''', 'intervals')

    def depth(self, index):
        return self._copper_depth if index == 0 else sum(self._depth_limits[index-1])*.5

    def plane(self, index):
        if index == 0:
            return self._copper
        result = self._cp.zeros(self.shape, self._cp.float32)
        if self._pixels.size:
            bottom, top = self._depth_limits[index-1]
            self._accumulate(((self._pixels.size+255)//256,), (256,),
                (self._pixels, self._lows, self._highs, numpy.int32(self._pixels.size),
                 numpy.float64(bottom), numpy.float64(top), result))
        return result


class GPUCompositeDepthPlanes(typing.Sequence):
    """Render independent material intervals without filling vacuum between them."""
    _on_gpu = True

    def __init__(self, layers):
        self.layers = tuple(layers)

    def __len__(self):
        return sum(len(layer) for layer in self.layers)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        for layer in self.layers:
            if 0 <= index < len(layer):
                return layer[index]
            index -= len(layer)
        raise IndexError(index)

    def render(self, *, return_device=False, **settings):
        cp = self.layers[0]._cp
        result = cp.zeros(self.layers[0].shape, dtype=cp.float32)
        for layer in self.layers:
            result += layer.render(return_device=True, **settings)
        return result if return_device else cp.asnumpy(result)
