"""Optional CUDA implementation of the vertical triangle surface rasterizer."""

from __future__ import annotations

import typing

import numpy
import numpy.typing


_SOURCE = r'''
__device__ void extreme(double* address, double value, bool maximum) {
    unsigned long long* bits = (unsigned long long*)address;
    unsigned long long old = atomicCAS(bits, 0ULL, 0ULL), assumed;
    do {
        assumed = old;
        double previous = __longlong_as_double(assumed);
        if (maximum ? value <= previous : value >= previous) return;
        old = atomicCAS(bits, assumed, __double_as_longlong(value));
    } while (old != assumed);
}
extern "C" __global__ void surfaces(
    const double* triangles, const double* x, const double* y,
    const int* bounds, int width, double* lower, double* upper) {
    int i = blockIdx.x;
    const double* t = triangles + i * 9;
    const int* box = bounds + i * 4;
    int box_width = box[1] - box[0];
    int count = box_width * (box[3] - box[2]);
    double ax = t[3]-t[0], ay = t[4]-t[1], az = t[5]-t[2];
    double bx = t[6]-t[0], by = t[7]-t[1], bz = t[8]-t[2];
    double inverse = 1.0 / (ax*by-ay*bx);
    for (int pixel = threadIdx.x; pixel < count; pixel += blockDim.x) {
        int row = box[2] + pixel / box_width;
        int col = box[0] + pixel % box_width;
        double dx = x[col]-t[0], dy = y[row]-t[1];
        double a = (dx*by-dy*bx)*inverse;
        double b = (dy*ax-dx*ay)*inverse;
        if (a >= -1e-8 && b >= -1e-8 && a+b <= 1.0+1e-8) {
            double z = t[2] + a*az + b*bz;
            extreme(lower + row*width+col, z, false);
            extreme(upper + row*width+col, z, true);
        }
    }
}
'''


class GPUSurfaceRasterizer:
    def __init__(self, triangles: numpy.typing.NDArray[numpy.float64]) -> None:
        # Import lazily: CuPy is an optional dependency and CPU remains usable
        # on machines without CUDA or when a driver cannot initialize.
        import cupy

        if cupy.cuda.runtime.getDeviceCount() == 0:
            raise RuntimeError("No CUDA device is available")
        self._cupy: typing.Any = cupy
        self._triangles = cupy.asarray(triangles)
        self._kernel = cupy.RawKernel(_SOURCE, "surfaces", options=("--fmad=false",))
        # Compile during sample initialization, not during the first zoom.
        self._kernel.compile()

    def surface_maps(
        self,
        x_nm: numpy.typing.NDArray[numpy.float64],
        y_nm: numpy.typing.NDArray[numpy.float64],
        candidates: numpy.typing.NDArray[numpy.int64],
        bounds: numpy.typing.NDArray[numpy.int32],
    ) -> tuple[numpy.typing.NDArray[numpy.float32], numpy.typing.NDArray[numpy.float32]]:
        cp = self._cupy
        lower = cp.full((y_nm.size, x_nm.size), cp.inf, dtype=cp.float64)
        upper = cp.full_like(lower, -cp.inf)
        if candidates.size:
            triangles = self._triangles[cp.asarray(candidates)]
            self._kernel(
                (candidates.size,), (128,),
                (triangles, cp.asarray(x_nm), cp.asarray(y_nm), cp.asarray(bounds),
                 numpy.int32(x_nm.size), lower, upper),
            )
        lower[~cp.isfinite(lower)] = cp.nan
        upper[~cp.isfinite(upper)] = cp.nan
        return cp.asnumpy(lower.astype(cp.float32)), cp.asnumpy(upper.astype(cp.float32))
