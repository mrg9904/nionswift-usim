"""Shared active sample rotations: lab R_y(TY) @ R_x(TX)."""
import math
import numpy as np


def stage_rotation(tilt):
    if not np.isfinite([tilt.x, tilt.y]).all():
        raise ValueError('Stage tilt must be finite')
    cx, sx, cy, sy = math.cos(tilt.x), math.sin(tilt.x), math.cos(tilt.y), math.sin(tilt.y)
    return np.array([[cy, sy*sx, sy*cx], [0., cx, -sx], [-sy, cy*sx, cy*cx]])


def rotate_triangles(triangles, tilt, pivot):
    return (triangles-np.asarray(pivot)) @ stage_rotation(tilt).T + pivot


def prepare_sample(sample, instrument):
    setter = getattr(sample, 'set_stage_tilt', None)
    if setter is not None:
        setter(instrument.GetVal2D('stage_tilt_rad'))
