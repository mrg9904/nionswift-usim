"""Exact tilted mesh projection, rotating only triangles near the current FoV."""
from collections import OrderedDict
import numpy as np
from nion.utils import Geometry
from nion.usim_device import SampleGeometry, SurfaceRasterizer
from nion.usim_device import SimulationSettings


class TiltedSurfaceRasterizer:
    def __init__(self, triangles, pivot):
        self.triangles = np.asarray(triangles, dtype=np.float64)
        self.pivot = np.asarray(pivot, dtype=float)
        self.minimum = self.triangles[:, :, :2].min(axis=1)
        self.maximum = self.triangles[:, :, :2].max(axis=1)
        self.z_bounds = (self.triangles[:, :, 2].min(), self.triangles[:, :, 2].max())
        self.tilt = Geometry.FloatPoint()
        self.rotation = np.eye(3)
        self.cache = OrderedDict()

    def set_tilt(self, tilt):
        if tilt != self.tilt:
            self.rotation = SampleGeometry.stage_rotation(tilt)
            self.tilt = tilt
            self.cache.clear()

    def surface_maps(self, x_nm, y_nm, *, on_gpu=False):
        key = (x_nm[0], x_nm[-1], y_nm[0], y_nm[-1], x_nm.size, y_nm.size)
        rasterizer = self.cache.get(key)
        if rasterizer is None:
            # At fixed original Z, lab XY is an affine function of original
            # XY. Invert it at FoV corners and both Z bounds to conservatively
            # enclose every possible intersecting source triangle.
            if abs(np.linalg.det(self.rotation[:2, :2])) < 1e-10:
                candidates = np.arange(len(self.triangles))
            else:
                corners = np.array([[x, y] for x in (x_nm[0], x_nm[-1]) for y in (y_nm[0], y_nm[-1])])
                inverse = np.linalg.inv(self.rotation[:2, :2])
                projected = np.concatenate([(corners-self.pivot[:2]
                    -self.rotation[:2, 2]*(z-self.pivot[2])) @ inverse.T + self.pivot[:2]
                    for z in self.z_bounds])
                lower, upper = projected.min(axis=0)-1e-5, projected.max(axis=0)+1e-5
                candidates = np.flatnonzero(((self.maximum >= lower) & (self.minimum <= upper)).all(axis=1))
            triangles = (self.triangles[candidates]-self.pivot) @ self.rotation.T+self.pivot
            rasterizer = SurfaceRasterizer.SurfaceRasterizer(triangles)
            self.cache[key] = rasterizer
            if len(self.cache) > 4:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(key)
        return rasterizer.surface_maps(x_nm, y_nm, on_gpu=on_gpu)

    def uses_gpu(self, shape):
        if SimulationSettings.STL_SURFACE_BACKEND == 'cpu':
            return False
        if SimulationSettings.STL_SURFACE_BACKEND != 'gpu' and np.prod(shape) < SimulationSettings.STL_GPU_MINIMUM_PIXELS:
            return False
        try:
            import cupy as cp
            return bool(cp.cuda.runtime.getDeviceCount())
        except Exception:
            return False
