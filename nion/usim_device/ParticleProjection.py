"""Interpolate all sparse particle projections with one shared ray map.

Overlapping particles remain separate channels; no label overwrites another.
The four bilinear neighbours match scipy's constant-boundary interpolation.
"""
import numpy as np
from nion.usim_device import SimulationSettings


def mapped_particles(pieces, source_shape, coordinates, binning=(1, 1)):
    xp = np
    if SimulationSettings.RONCHIGRAM_BACKEND == 'gpu' or (SimulationSettings.RONCHIGRAM_BACKEND == 'auto' and
            coordinates[0].size >= SimulationSettings.RONCHIGRAM_GPU_MINIMUM_PIXELS):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount():
                return _mapped_particles(pieces, source_shape, coordinates, binning, cp)
        except Exception as error:
            import logging
            logging.getLogger(__name__).warning('Sparse ray mapping CUDA unavailable; using CPU: %s', error)
    return _mapped_particles(pieces, source_shape, coordinates, binning, xp)


def _mapped_particles(pieces, source_shape, coordinates, binning, xp):
    height, width = source_shape
    by, bx = binning
    if by == 1:
        bx = 1  # Match CameraSimulator's historical binning convention.
    shape = height//by, width//bx
    pixels, identifiers, values = [], [], []
    for identifier, region, lower, upper in pieces:
        thickness = np.where(np.isfinite(lower) & np.isfinite(upper), np.maximum(upper-lower, 0.), 0.)
        attenuation = -100*np.expm1(-thickness/SimulationSettings.RONCHIGRAM_TRANSMISSION_LENGTH_NM)
        yy, xx = np.nonzero(attenuation)
        sy, sx = yy+region[0].start, xx+region[1].start
        valid = (sy < shape[0]*by) & (sx < shape[1]*bx)
        pixels.append((sy[valid]//by)*shape[1]+sx[valid]//bx)
        identifiers.append(np.full(valid.sum(), identifier, np.int32))
        values.append(attenuation[yy[valid], xx[valid]])
    if not pixels:
        return {}, np.zeros(coordinates[0].shape, bool), 'gpu' if xp is not np else 'cpu'
    pixels, identifiers, values = np.concatenate(pixels), np.concatenate(identifiers), np.concatenate(values)
    order = np.argsort(pixels, kind='stable')
    pixels, identifiers, values = pixels[order], identifiers[order], values[order]
    pixels, identifiers, values = (xp.asarray(a) for a in (pixels, identifiers, values))
    cy, cx = (xp.asarray(c).ravel() for c in coordinates)
    inside = (cy >= 0) & (cy <= shape[0]-1) & (cx >= 0) & (cx <= shape[1]-1)
    targets = xp.flatnonzero(inside)
    fy, fx = xp.floor(cy[inside]).astype(int), xp.floor(cx[inside]).astype(int)
    dy, dx = cy[inside]-fy, cx[inside]-fx
    keys, contributions = [], []
    for oy, ox, weight in ((0, 0, (1-dy)*(1-dx)), (0, 1, (1-dy)*dx), (1, 0, dy*(1-dx)), (1, 1, dy*dx)):
        neighbour = xp.minimum(fy+oy, shape[0]-1)*shape[1]+xp.minimum(fx+ox, shape[1]-1)
        start, end = xp.searchsorted(pixels, neighbour, 'left'), xp.searchsorted(pixels, neighbour, 'right')
        counts = end-start
        selected = xp.repeat(xp.arange(len(targets)), counts)
        if not len(selected):
            continue
        entry = xp.repeat(start, counts)+xp.arange(int(counts.sum()))-xp.repeat(xp.cumsum(counts)-counts, counts)
        keys.append(identifiers[entry].astype(np.int64)*len(cy)+targets[selected])
        contributions.append(values[entry]*weight[selected])
    if not keys:
        return {}, (xp.asnumpy(inside) if xp is not np else inside).reshape(coordinates[0].shape), 'gpu' if xp is not np else 'cpu'
    keys, contributions = xp.concatenate(keys), xp.concatenate(contributions)
    order = xp.argsort(keys)
    keys, contributions = keys[order], contributions[order]
    unique, starts = xp.unique(keys, return_index=True)
    # reduceat is not provided by every CuPy version; cumulative sums give
    # the same segmented reduction and work on both backends.
    cumulative = xp.concatenate((xp.zeros(1, contributions.dtype), xp.cumsum(contributions)))
    ends = xp.concatenate((starts[1:], xp.asarray([len(contributions)])))
    values = cumulative[ends]-cumulative[starts]
    valid = values > 1e-4
    unique, values = unique[valid], values[valid]
    ids = unique//len(cy)
    result = {}
    distinct = xp.asnumpy(xp.unique(ids)) if xp is not np else np.unique(ids)
    for identifier in distinct:
        mask = ids == identifier
        indexes, attenuation = unique[mask]%len(cy), values[mask].astype(np.float32)
        if xp is not np:
            indexes, attenuation = xp.asnumpy(indexes), xp.asnumpy(attenuation)
        result[int(identifier)] = indexes, attenuation
    return result, (xp.asnumpy(inside) if xp is not np else inside).reshape(coordinates[0].shape), 'gpu' if xp is not np else 'cpu'
