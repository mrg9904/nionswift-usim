"""CIF-driven TEM Kikuchi geometry; intensities are illustrative, not dynamical."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import math
from pathlib import Path

import gemmi
import numpy as np
from scipy import constants
from scipy.ndimage import gaussian_filter
from nion.usim_device import SimulationSettings
from nion.usim_device import SampleGeometry
from nion.utils import Geometry


@dataclass(frozen=True)
class Crystal:
    cell_angstrom: np.ndarray  # columns are real-space a, b, c
    fractional_positions: np.ndarray
    elements: tuple[str, ...]
    occupancies: np.ndarray
    u_iso: np.ndarray
    source: str

    @property
    def reciprocal(self) -> np.ndarray:
        """Cycles/angstrom (no 2*pi)."""
        return np.linalg.inv(self.cell_angstrom).T

    def structure_factor(self, hkl: tuple[int, int, int]) -> complex:
        g2 = float(np.sum((self.reciprocal @ hkl)**2))
        factors = np.array([gemmi.Element(e).c4322.calculate_sf(g2/4) for e in self.elements])
        return complex(np.sum(self.occupancies * factors * np.exp(-2*np.pi**2*self.u_iso*g2)
                              * np.exp(2j*np.pi*(self.fractional_positions @ hkl))))


@lru_cache(maxsize=8)
def load_crystal(path: str) -> Crystal:
    """Expand CIF symmetry per site, preserving mixed occupancies at common positions."""
    document = gemmi.cif.read_file(str(Path(path).resolve()))
    for block in document:
        if block.find_value('_cell_length_a') and block.find_values('_atom_site_fract_x'):
            structure = gemmi.make_small_structure_from_block(block)
            if structure.cell.volume <= 0:
                raise ValueError('CIF must contain a valid unit cell')
            sites = structure.get_all_unit_cell_sites()
            if not sites or any(s.element.atomic_number == 0 or s.element.c4322 is None for s in sites):
                raise ValueError('CIF must contain supported atomic elements')
            return Crystal(np.array(structure.cell.orth.mat),
                np.array([tuple(s.fract) for s in sites]) % 1,
                tuple(s.element.name for s in sites), np.array([s.occ for s in sites]),
                np.array([max(0., s.u_iso) for s in sites]), str(Path(path).resolve()))
    raise ValueError('No crystal cell and fractional atomic coordinates found in CIF')


def wavelength_angstrom(voltage_v: float) -> float:
    if not math.isfinite(voltage_v) or voltage_v <= 0:
        raise ValueError('Accelerating voltage must be finite and positive')
    energy = constants.e * voltage_v
    return constants.h / math.sqrt(2*constants.m_e*energy*(1+energy/(2*constants.m_e*constants.c**2))) * 1e10


def orientation_matrix(crystal: Crystal, zone_axis=(0, 0, 1), tx_rad=0., ty_rad=0., *, sample_rotation=None) -> np.ndarray:
    """Align [uvw] with z, projected a with x; active lab tilts R_y(TY) @ R_x(TX)."""
    z = crystal.cell_angstrom @ np.asarray(zone_axis, dtype=float)
    if not np.all(np.isfinite(z)) or np.linalg.norm(z) == 0 or not np.isfinite([tx_rad, ty_rad]).all():
        raise ValueError('Zone axis must be nonzero and tilts finite')
    z /= np.linalg.norm(z)
    x = crystal.cell_angstrom[:, 0] - z*np.dot(z, crystal.cell_angstrom[:, 0])
    if np.linalg.norm(x) < 1e-10:
        x = crystal.cell_angstrom[:, 1] - z*np.dot(z, crystal.cell_angstrom[:, 1])
    x /= np.linalg.norm(x)
    initial = np.stack([x, np.cross(z, x), z])
    if sample_rotation is not None:
        rotation = np.asarray(sample_rotation, dtype=float)
        if (rotation.shape != (3, 3) or not np.isfinite(rotation).all() or
                not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8) or
                not np.isclose(np.linalg.det(rotation), 1.)):
            raise ValueError('Sample rotation must be a proper orthonormal matrix')
        initial = rotation @ initial
    return SampleGeometry.stage_rotation(Geometry.FloatPoint(x=tx_rad, y=ty_rad)) @ initial


@dataclass(frozen=True)
class Band:
    hkl: tuple[int, int, int]
    normal: np.ndarray
    d_angstrom: float
    theta_b_rad: float
    weight: float


@lru_cache(maxsize=8)
def _reflectors(path: str, d_min_angstrom: float):
    crystal = load_crystal(path)
    # |h_i| <= |a_i|/d_min, valid for non-orthogonal cells too.
    limits = np.ceil(np.linalg.norm(crystal.cell_angstrom, axis=0)/d_min_angstrom).astype(int)
    reflections = []
    for hkl in product(*(range(-int(n), int(n)+1) for n in limits)):
        # Keep exactly one member of each Friedel pair, not one symmetry family.
        if hkl == (0, 0, 0) or next(v for v in hkl if v) < 0:
            continue
        g = crystal.reciprocal @ hkl
        length = np.linalg.norm(g)
        if length > 1/d_min_angstrom + 1e-10:
            continue
        strength = abs(crystal.structure_factor(hkl))**2
        reflections.append((hkl, g/length, 1/length, strength))
    maximum = max((r[3] for r in reflections), default=0)
    return tuple(r for r in reflections if r[3] > maximum*1e-5)


def bands(crystal: Crystal, voltage_v: float, zone_axis=(0, 0, 1), tx_rad=0., ty_rad=0.,
          max_angle_rad=.1, d_min_angstrom=.75, max_bands=160, *, sample_rotation=None) -> list[Band]:
    wavelength = wavelength_angstrom(voltage_v)
    orientation = orientation_matrix(crystal, zone_axis, tx_rad, ty_rad, sample_rotation=sample_rotation)
    visible = []
    for hkl, normal, spacing, strength in _reflectors(crystal.source, d_min_angstrom):
        sin_b = wavelength/(2*spacing)
        if sin_b >= 1:
            continue
        normal = orientation @ normal
        theta_b = math.asin(sin_b)
        if abs(math.asin(float(np.clip(normal[2], -1, 1)))) > max_angle_rad + theta_b:
            continue
        visible.append((hkl, normal, spacing, theta_b, strength))
    visible.sort(key=lambda r: (-r[4], r[0]))
    maximum = max((r[4] for r in visible), default=1)
    return [Band(h, n, d, b, math.sqrt(s/maximum)) for h, n, d, b, s in visible[:max_bands]]


def render_lines(band_list: list[Band], x_rad: np.ndarray, y_rad: np.ndarray,
                 line_width_rad=.00035) -> np.ndarray:
    """Project paired Bragg cones n.g_hat=+/-sin(theta_B), including curvature.

    Positive/negative Gaussian boundaries are display contrast, not a prediction
    of excess/deficiency intensity from dynamical diffraction.
    """
    x, y = np.meshgrid(np.tan(x_rad), np.tan(y_rad))
    norm = np.sqrt(1+x*x+y*y)
    result = np.zeros(x.shape, dtype=np.float32)
    for band in band_list:
        distance = (band.normal[0]*x + band.normal[1]*y + band.normal[2])/norm
        half_width = math.sin(band.theta_b_rad)
        for sign in (-1, 1):
            delta = (distance - sign*half_width)/line_width_rad
            selected = np.abs(delta) < 4
            result[selected] += sign*band.weight*np.exp(-.5*delta[selected]**2)
    # Fixed transfer avoids renormalizing a thin specimen to full contrast.
    return np.tanh(result)


def thickness_contrast(thickness_nm, *, xp=np):
    """Build-up at low thickness, then attenuation of ordered contrast."""
    t = xp.maximum(0., thickness_nm)
    return .35 * (-xp.expm1(-t/25.)) * xp.exp(-t/150.)


def transmission(thickness_nm):
    """Illustrative bright-field attenuation, with vacuum normalized to one."""
    return np.exp(-np.maximum(0., thickness_nm)/SimulationSettings.RONCHIGRAM_TRANSMISSION_LENGTH_NM)


def diffusion_sigma_rad(thickness_nm):
    """Angular random-walk broadening; sigma scales with sqrt(path length)."""
    return SimulationSettings.KIKUCHI_BROADENING_RAD_AT_100_NM * np.sqrt(np.maximum(0., thickness_nm)/100.)


def compose_contrast(real_image, vacuum_level, line_pattern, pixel_angles_rad):
    """Combine existing aberration-mapped real image, diffuse halo and lines.

    Recover the projected thickness from its transmission. Blend a small
    bank of angularly broadened patterns to give thick and thin parts of
    the SAME defocused sphere different line widths. Gaussian smoothing
    preserves integrated signed line contrast; do not renormalize peaks.
    These display parameters are phenomenological, not a scattering solver.
    """
    pixel_angles = np.abs(np.asarray(pixel_angles_rad, dtype=float))
    if np.any(pixel_angles <= 0) or vacuum_level <= 0:
        raise ValueError('Positive pixel angles and vacuum intensity required')
    real = np.clip(np.asarray(real_image, dtype=float), 0, vacuum_level)
    thickness = -SimulationSettings.RONCHIGRAM_TRANSMISSION_LENGTH_NM * np.log(
        np.clip(real/vacuum_level, 1e-12, 1.))
    # Zero-filled rays outside the source image are not an opaque specimen.
    # Preserve those dark pixels rather than creating a diffuse pedestal.
    thickness = np.where(real > 0, thickness, 0.)
    diffuse_fraction = -np.expm1(-thickness/SimulationSettings.RONCHIGRAM_DIFFUSE_LENGTH_NM)
    halo = gaussian_filter(real, SimulationSettings.RONCHIGRAM_DIFFUSE_SIGMA_RAD/pixel_angles, mode='nearest')
    # Part of the unstructured scattering fills specimen contrast with a
    # smooth background. Vacuum stays at its original intensity.
    diffuse_background = halo + .3*(vacuum_level-halo)
    base = (1-diffuse_fraction)*real + diffuse_fraction*diffuse_background
    levels = [0., 10., 25., 50., 100.]
    while levels[-1] < thickness.max():
        levels.append(2*levels[-1])
    levels = np.asarray(levels)
    lower = np.clip(np.searchsorted(levels, thickness, side='right')-1, 0, len(levels)-2)
    upper = lower+1
    fraction = (thickness-levels[lower])/(levels[upper]-levels[lower])
    blurred = np.zeros(real.shape)
    for index, level in enumerate(levels):
        weight = np.where(lower == index, 1-fraction, 0) + np.where(upper == index, fraction, 0)
        if np.any(weight > 0):
            sigma = diffusion_sigma_rad(level)/pixel_angles
            band_image = gaussian_filter(line_pattern, sigma, mode='nearest') if level else line_pattern
            blurred += weight*band_image
    result = base + vacuum_level*thickness_contrast(thickness)*blurred
    return np.maximum(result, 0), {
        'projected_thickness_range_nm': [float(thickness.min()), float(thickness.max())],
        'diffuse_fraction_range': [float(diffuse_fraction.min()), float(diffuse_fraction.max())]}
