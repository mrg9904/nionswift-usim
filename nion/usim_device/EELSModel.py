"""Geometry-driven, synthetic compound-Poisson energy-loss model.

Independent inelastic events use tau=sum(t/lambda). Single-event kernels are
phenomenological plasmon/core-edge/power-law continuum shapes, not cross-section calculations.
Convolution is linear and truncated beyond the modeled energy range; missing
high-energy counts are never renormalized into the detector window.
"""
from dataclasses import dataclass, field
import math
import numpy as np
from scipy import signal, stats
from nion.utils import Geometry
from nion.usim_device import SimulationSettings


@dataclass(frozen=True)
class EELSMaterial:
    edges: tuple = ((68, 30), (855, 50), (872, 50))
    plasmon_eV: float = 20.0
    mean_free_path_nm: float = 100.0
    core_fraction: float = .03
    background_fraction: float = field(default_factory=lambda: SimulationSettings.EELS_BACKGROUND_FRACTION)
    background_exponent: float = field(default_factory=lambda: SimulationSettings.EELS_BACKGROUND_EXPONENT)
    background_transition_eV: float = field(default_factory=lambda: SimulationSettings.EELS_BACKGROUND_TRANSITION_EV)

    def validate(self):
        if not math.isfinite(self.mean_free_path_nm) or self.mean_free_path_nm <= 0:
            raise ValueError("EELS mean free path must be finite and positive")
        if not math.isfinite(self.plasmon_eV) or self.plasmon_eV <= 0:
            raise ValueError("EELS plasmon energy must be finite and positive")
        if not math.isfinite(self.core_fraction) or not 0 <= self.core_fraction <= 1:
            raise ValueError("EELS core fraction must be between zero and one")
        if (not math.isfinite(self.background_fraction) or self.background_fraction < 0 or
                self.background_fraction+self.core_fraction > 1):
            raise ValueError("EELS core and background fractions must be nonnegative and sum to at most one")
        if not math.isfinite(self.background_exponent) or self.background_exponent <= 1:
            raise ValueError("EELS background exponent must be finite and greater than one")
        if not math.isfinite(self.background_transition_eV) or self.background_transition_eV <= 0:
            raise ValueError("EELS background transition must be finite and positive")
        for energy, width in self.edges:
            if not math.isfinite(energy) or energy <= 0 or not math.isfinite(width) or width <= 0:
                raise ValueError("EELS edges require positive finite onset and width")


@dataclass(frozen=True)
class EELSLayer:
    thickness_nm: float
    material: EELSMaterial


def probe_sample_position(offset_m, fov_size_nm, center_nm, probe_position, rotation_rad=0.0):
    """Normalized scan position -> sample XY, matching uSim rotated scans."""
    x = (probe_position.x - .5) * fov_size_nm.width
    y = (probe_position.y - .5) * fov_size_nm.height
    cosine, sine = math.cos(rotation_rad), math.sin(rotation_rad)
    return Geometry.FloatPoint(
        x=(center_nm.x + cosine*x - sine*y)*1e-9 - offset_m.x,
        y=(center_nm.y + sine*x + cosine*y)*1e-9 - offset_m.y)


def powerlaw_background_cdf(energy, exponent, transition_eV):
    """Normalized continuous loss distribution; exact E^-r above transition.

    Below E0 use x^2 * ((r+3)-(r+2)*x), x=E/E0. Value and first
    derivative match the tail at E0; density vanishes at zero loss.
    Analytic normalization extends to infinity, independent of detector range.
    """
    if not math.isfinite(exponent) or exponent <= 1 or not math.isfinite(transition_eV) or transition_eV <= 0:
        raise ValueError("Power-law background requires exponent > 1 and positive transition")
    x = np.maximum(np.asarray(energy, dtype=float), 0)/transition_eV
    low = np.minimum(x, 1.)
    normalization = (exponent+6)/12 + 1/(exponent-1)
    low_integral = (exponent+3)*low**3/3 - (exponent+2)*low**4/4
    tail_integral = (exponent+6)/12 - np.expm1((1-exponent)*np.log(np.maximum(x, 1.)))/(exponent-1)
    return np.where(x <= 1, low_integral, tail_integral)/normalization


def single_event_kernel(material, energy):
    material.validate()
    plasmon = stats.norm.pdf(energy, loc=material.plasmon_eV, scale=math.sqrt(material.plasmon_eV))
    plasmon /= plasmon.sum()
    core = np.zeros_like(energy)
    for onset, width in material.edges:
        edge = stats.norm.cdf(energy, loc=onset, scale=width) * stats.powerlaw.pdf(4000-energy, 8, loc=0, scale=4000)
        total = edge.sum()
        if total > 0:
            core += edge / total
    core_fraction = 0.
    if core.sum() > 0:
        core /= core.sum()
        core_fraction = material.core_fraction
    # Integrate the continuum per quadrature bin. Never renormalize a long
    # power-law tail into the finite computational/detector window.
    background = np.zeros_like(energy)
    if material.background_fraction:
        step = float(energy[1]-energy[0])
        background = (powerlaw_background_cdf(energy+step/2, material.background_exponent, material.background_transition_eV)
                      - powerlaw_background_cdf(energy-step/2, material.background_exponent, material.background_transition_eV))
    return ((1-core_fraction-material.background_fraction)*plasmon + core_fraction*core
            + material.background_fraction*background)


def spectrum_probabilities(layers, channel_energies, dispersion_eV, zlp_sigma_eV=.5):
    """Return integrated probability per channel and geometry/scattering metadata.

    Channel calibrations describe centers. Internal quadrature <=0.25 eV avoids
    losing the ZLP at coarse dispersion. Entire-window normalization is forbidden.
    """
    channel_energies = np.asarray(channel_energies, dtype=float)
    if (channel_energies.ndim != 1 or not len(channel_energies) or
        not np.isfinite(channel_energies).all() or
        not math.isfinite(dispersion_eV) or dispersion_eV <= 0 or
        not math.isfinite(zlp_sigma_eV) or zlp_sigma_eV <= 0):
        raise ValueError("EELS requires finite channel energies and positive dispersion/resolution")
    tau = 0.0
    thickness = 0.0
    weighted = []
    background_depth = 0.
    for layer in layers:
        layer.material.validate()
        if not math.isfinite(layer.thickness_nm) or layer.thickness_nm < 0:
            raise ValueError("EELS thickness must be finite and nonnegative")
        thickness += layer.thickness_nm
        depth = layer.thickness_nm / layer.material.mean_free_path_nm
        tau += depth
        background_depth += depth*layer.material.background_fraction
        if depth:
            weighted.append((depth, layer.material))
    if not math.isfinite(tau):
        raise ValueError("EELS optical thickness must be finite")
    p0 = math.exp(-tau)
    lower = channel_energies - dispersion_eV/2
    upper = channel_energies + dispersion_eV/2
    probabilities = p0 * (stats.norm.cdf(upper, scale=zlp_sigma_eV) - stats.norm.cdf(lower, scale=zlp_sigma_eV))
    metadata = {"thickness_nm": thickness, "t_over_lambda": tau,
                "zero_loss_fraction": p0, "model": "synthetic_compound_poisson_powerlaw_v2",
                "background_event_optical_depth": background_depth,
                "background_event_probability": -math.expm1(-background_depth),
                "background_components": [{"t_over_lambda": depth,
                    "single_event_fraction": material.background_fraction,
                    "exponent": material.background_exponent,
                    "transition_eV": material.background_transition_eV}
                    for depth, material in weighted if material.background_fraction]}
    if tau:
        step = min(.25, dispersion_eV)
        limit = max(4000., float(upper.max()) + 10*zlp_sigma_eV)
        energy = np.arange(int(math.ceil(limit/step)) + 1) * step
        kernel = sum(depth/tau * single_event_kernel(material, energy) for depth, material in weighted)
        # The order count grows with thickness; feature.plurality no longer caps it.
        max_order = int(stats.poisson.ppf(1-1e-10, tau))
        current = np.zeros_like(energy)
        current[0] = 1.
        loss = np.zeros_like(energy)
        for order in range(1, max_order+1):
            current = np.maximum(0, signal.fftconvolve(current, kernel)[:len(energy)])
            loss += stats.poisson.pmf(order, tau) * current
        radius = max(1, int(math.ceil(8*zlp_sigma_eV/step)))
        response = stats.norm.pdf(np.arange(-radius, radius+1)*step, scale=zlp_sigma_eV)
        response /= response.sum()
        broadened = np.maximum(0, signal.fftconvolve(loss, response))
        bin_edges = (np.arange(len(broadened)+1) - radius - .5)*step
        cumulative = np.r_[0., np.cumsum(broadened)]
        probabilities += np.interp(upper, bin_edges, cumulative) - np.interp(lower, bin_edges, cumulative)
        metadata.update(max_scattering_order=max_order, poisson_tail_probability=float(stats.poisson.sf(max_order, tau)))
    else:
        metadata.update(max_scattering_order=0, poisson_tail_probability=0.)
    probabilities = np.maximum(probabilities, 0)
    metadata["detector_window_fraction"] = float(probabilities.sum())
    return probabilities, metadata
