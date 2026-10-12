"""Bounded-memory spectrum imaging with the existing compound-Poisson model.

This is the uSim batch backend, not a hardware synchronization implementation.
Spectra use material optical-depth bins of 0.01, with electron shot noise.
The consumer obtains spectra, not a simulator thickness map for analysis.
"""
import math
import time
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy import ndimage
from nion.utils import Geometry
from nion.usim_device import EELSModel, SampleGeometry


@lru_cache(maxsize=4096)
def _cached_spectrum(code, materials, offset, dispersion, channels, slit):
    energies = offset+np.arange(channels)*dispersion
    layers = [EELSModel.EELSLayer(value*.01*material.mean_free_path_nm, material)
              for value, material in zip(code, materials) if value > 0]
    probabilities, _ = EELSModel.spectrum_probabilities(layers, energies, dispersion, .5/slit)
    result = probabilities.astype(np.float32)
    result.flags.writeable = False
    return result


def _layer_maps(sample, instrument, parameters, check_cancelled=lambda: None):
    n = int(parameters['pixel_size'][0])
    rotation = float(parameters.get('rotation_rad', 0.))
    pixel = float(parameters['fov_nm'])/n
    size = int(math.ceil(n*(abs(math.cos(rotation))+abs(math.sin(rotation)))))
    center = Geometry.FloatPoint.make(parameters.get('center_nm', (0., 0.)))
    offset = instrument.value_manager.actual_offset_m
    args = (offset, Geometry.FloatSize(size*pixel, size*pixel), Geometry.FloatPoint(),
            center, Geometry.IntSize(size, size))
    SampleGeometry.prepare_sample(sample, instrument)
    # Cathode specimens expose their separate support and crystal projections.
    # The same cached rasterizers are used by HAADF and Ronchigram.
    if hasattr(sample, '_maps') and hasattr(sample, '_carbon_mask'):
        maps = sample._maps(*args)
        lo, hi = maps[:2]
        support = sample._thickness(lo, hi)
        x = center.x-offset.x*1e9+(np.arange(size)+.5-size/2)*pixel
        y = center.y-offset.y*1e9+(np.arange(size)+.5-size/2)*pixel
        carbon = sample._carbon_mask(x, y, lo, hi)
        crystal = np.zeros((size, size), float)
        if len(maps) == 3:
            for _, region, lower, upper in maps[2]:
                crystal[region] += sample._thickness(lower, upper)
        else:
            crystal = sample._thickness(maps[2], maps[3])
        materials = (
            EELSModel.EELSMaterial(edges=((284., 15.),), plasmon_eV=24., mean_free_path_nm=150.),
            EELSModel.EELSMaterial(edges=((75., 20.), (932., 30.), (952., 30.)), plasmon_eV=19.),
            EELSModel.EELSMaterial())
        fields = [np.where(carbon, support, 0.), np.where(~carbon, support, 0.), crystal]
        yy, xx = np.mgrid[:n, :n]
        xx, yy = xx+.5-n/2, yy+.5-n/2
        cosine, sine = math.cos(rotation), math.sin(rotation)
        coordinates = [sine*xx+cosine*yy+(size-1)/2, cosine*xx-sine*yy+(size-1)/2]
        return [(ndimage.map_coordinates(field, coordinates, order=1, mode='nearest'), material)
                for field, material in zip(fields, materials)]
    # Exact point queries keep the backend usable for the other uSim samples.
    fields = {}
    for row in range(n):
        check_cancelled()
        for column in range(n):
            position = EELSModel.probe_sample_position(offset, Geometry.FloatSize(parameters['fov_nm'], parameters['fov_nm']),
                center, Geometry.FloatPoint((row+.5)/n, (column+.5)/n), rotation)
            for layer in sample.eels_layers_at(position):
                if layer.material not in fields:
                    fields[layer.material] = np.zeros((n, n))
                fields[layer.material][row, column] += layer.thickness_nm
    return [(field, material) for material, field in fields.items()]


def acquire(camera, parameters, *, output_path, energy_channel_count=None, check_cancelled=lambda: None):
    started = time.monotonic()
    instrument = camera.instrument
    n, m = parameters['pixel_size']
    if n != m or n < 1 or parameters['pixel_time_us'] <= 0:
        raise ValueError('Batch EELS SI requires a positive square scan')
    check_cancelled()
    layers = _layer_maps(instrument.scan_data_generator.sample, instrument, parameters, check_cancelled)
    sensor_channels = instrument.camera_sensor_dimensions('eels')[1]
    channels = sensor_channels if energy_channel_count is None else energy_channel_count
    if not isinstance(channels, int) or not 2 <= channels <= sensor_channels:
        raise ValueError('Invalid energy channel count')
    dispersion = instrument.value_manager.energy_per_channel_eV
    energies = instrument.value_manager.energy_offset_eV+np.arange(channels)*dispersion
    codes = np.column_stack([np.rint(field.ravel()/material.mean_free_path_nm/.01).astype(np.int32)
                             for field, material in layers]) if layers else np.zeros((n*m, 0), np.int32)
    unique, inverse = np.unique(codes, axis=0, return_inverse=True)
    bank = np.empty((len(unique), channels), np.float32)
    slit = 10 if instrument.value_manager.is_slit_in else 1
    materials = tuple(material for _, material in layers)
    for index, code in enumerate(unique):
        check_cancelled()
        bank[index] = _cached_spectrum(tuple(map(int, code)), materials, float(energies[0]),
                                       float(dispersion), channels, slit)
    gain = instrument.counts_per_electron
    counts = camera.get_total_counts(parameters['pixel_time_us']*1e-6)/slit
    if instrument.value_manager.is_blanked:
        counts = 0.
    output = (np.empty((n, m, channels), dtype=np.float32) if output_path is None else
              np.lib.format.open_memmap(Path(output_path), mode='w+', dtype=np.float32,
                                        shape=(n, m, channels)))
    metadata = dict(backend='usim_batched_compound_poisson',
        shape=[n, m, channels], pixel_time_us=parameters['pixel_time_us'],
        effective_defocus_m=instrument.GetVal('C10')-instrument.GetVal('stage_z_m'),
        material_optical_depth_quantization=.01, maximum_tau_rounding_error=.005*len(layers),
        geometry_sampling='bilinear lab projection at scan pixel pitch for cathode models',
        probe_model='point probe, matching the live EELS camera; defocus is recorded but does not broaden the probe',
        counts_per_electron=gain, energy_offset_eV=float(energies[0]), dispersion_eV=float(dispersion),
        spectrum_image_saved=output_path is not None, status='acquiring', completed_rows=0)
    sidecar = Path(output_path).with_suffix('.json') if output_path is not None else None
    try:
        for row in range(0, n, 8):
            check_cancelled()
            end = min(row+8, n)
            block = bank[inverse[row*m:end*m]].reshape(end-row, m, channels)*counts
            if camera.noise.enabled:
                block = camera.noise._rng.poisson(np.maximum(block, 0)/gain)*gain
            output[row:end] = block
            metadata['completed_rows'] = end
            if sidecar is not None:
                sidecar.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        metadata['status'] = 'complete'
    except BaseException:
        metadata['status'] = 'incomplete'
        raise
    finally:
        if isinstance(output, np.memmap):
            output.flush()
        metadata['elapsed_seconds'] = time.monotonic()-started
        if sidecar is not None:
            sidecar.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return output, energies, metadata
