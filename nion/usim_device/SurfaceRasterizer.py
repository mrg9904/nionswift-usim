"""Rasterize vertical mesh intersections without a ray spatial-index query."""

from __future__ import annotations

import numpy
import numpy.typing
import logging
import typing

from nion.usim_device import SimulationSettings


class SurfaceRasterizer:
    """Evaluate triangle surfaces at pixel centers, retaining both Z extrema.

    This is the same vertical-ray geometry used by STLDepthSample, rather
    than a resized projection or a voxel approximation. Vertical faces do
    not intersect a downward ray; all other triangles are interpolated in
    their projected XY footprint.
    """

    def __init__(self, triangles: numpy.typing.NDArray[numpy.float64]) -> None:
        triangles = numpy.asarray(triangles, dtype=numpy.float64)
        edge_a = triangles[:, 1] - triangles[:, 0]
        edge_b = triangles[:, 2] - triangles[:, 0]
        determinant = edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
        keep = determinant != 0.0
        self._triangles = triangles[keep]
        self._edge_a = edge_a[keep]
        self._edge_b = edge_b[keep]
        self._inverse_determinant = 1.0 / determinant[keep]
        self._minimum = self._triangles[:, :, :2].min(axis=1)
        self._maximum = self._triangles[:, :, :2].max(axis=1)
        self._gpu: typing.Any = None
        if SimulationSettings.STL_SURFACE_BACKEND != "cpu":
            try:
                from nion.usim_device.GPUSurfaceRasterizer import GPUSurfaceRasterizer
                self._gpu = GPUSurfaceRasterizer(self._triangles)
            except ImportError:
                pass
            except Exception:
                logging.warning("uSim CUDA surface rasterizer unavailable; using CPU", exc_info=True)

    def surface_maps(
        self,
        x_nm: numpy.typing.NDArray[numpy.float64],
        y_nm: numpy.typing.NDArray[numpy.float64],
        *, on_gpu: bool = False,
    ) -> typing.Tuple[typing.Any, typing.Any]:
        # Restrict work to triangles whose bounding boxes contain pixel
        # centers. searchsorted also clips triangles outside the current FoV.
        left = numpy.searchsorted(x_nm, self._minimum[:, 0], side="left")
        right = numpy.searchsorted(x_nm, self._maximum[:, 0], side="right")
        top = numpy.searchsorted(y_nm, self._minimum[:, 1], side="left")
        bottom = numpy.searchsorted(y_nm, self._maximum[:, 1], side="right")
        candidates = numpy.flatnonzero((left < right) & (top < bottom))
        if self._gpu is not None and (
            SimulationSettings.STL_SURFACE_BACKEND == "gpu"
            or x_nm.size * y_nm.size >= SimulationSettings.STL_GPU_MINIMUM_PIXELS
        ):
            bounds = numpy.column_stack((left[candidates], right[candidates], top[candidates], bottom[candidates])).astype(numpy.int32)
            try:
                return self._gpu.surface_maps(x_nm, y_nm, candidates, bounds, on_gpu=on_gpu)
            except Exception:
                self._gpu = None
                logging.warning("uSim CUDA surface calculation failed; using CPU", exc_info=True)
        lower = numpy.full((y_nm.size, x_nm.size), numpy.inf, dtype=numpy.float64)
        upper = numpy.full_like(lower, -numpy.inf)
        for index in candidates:
            rows = slice(top[index], bottom[index])
            columns = slice(left[index], right[index])
            origin = self._triangles[index, 0]
            dx = x_nm[columns][None, :] - origin[0]
            dy = y_nm[rows][:, None] - origin[1]
            edge_a = self._edge_a[index]
            edge_b = self._edge_b[index]
            inverse = self._inverse_determinant[index]
            a = (dx * edge_b[1] - dy * edge_b[0]) * inverse
            b = (dy * edge_a[0] - dx * edge_a[1]) * inverse
            inside = (a >= -1e-8) & (b >= -1e-8) & (a + b <= 1.0 + 1e-8)
            height = origin[2] + a * edge_a[2] + b * edge_b[2]
            numpy.minimum(lower[rows, columns], height, out=lower[rows, columns], where=inside)
            numpy.maximum(upper[rows, columns], height, out=upper[rows, columns], where=inside)
        missing = ~numpy.isfinite(lower) | ~numpy.isfinite(upper)
        lower[missing] = numpy.nan
        upper[missing] = numpy.nan
        return lower.astype(numpy.float32), upper.astype(numpy.float32)

    def uses_gpu(self, shape: typing.Tuple[int, int]) -> bool:
        return self._gpu is not None and (
            SimulationSettings.STL_SURFACE_BACKEND == "gpu"
            or shape[0] * shape[1] >= SimulationSettings.STL_GPU_MINIMUM_PIXELS
        )
