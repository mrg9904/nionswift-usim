"""Generate deterministic sphere EELS data and a review figure.

Run --data-only in nionswift-dev. Rendering needs Matplotlib; --render-only
can run in a separate plotting environment without Nion installed.
"""
import argparse
import csv
from pathlib import Path
import numpy as np


OUTPUT = Path(__file__).resolve().parent / 'eels_thickness_results'


def generate(directory):
    from dataclasses import replace
    from nion.usim_device import EELSModel, SampleSimulator
    from nion.utils import Geometry
    sphere = SampleSimulator.SphericalParticleSample(1000)
    energy = np.arange(-20., 1400., .5)
    radii = np.asarray([0., 20., 40., 49., 55.])
    spectra, thickness, optical, zlp, captured = [], [], [], [], []
    without_background = []
    for radius in radii:
        layers = sphere.eels_layers_at(Geometry.FloatPoint(x=radius*1e-9, y=0))
        spectrum, info = EELSModel.spectrum_probabilities(layers, energy, .5)
        spectra.append(spectrum)
        old_layers = [EELSModel.EELSLayer(layer.thickness_nm, replace(layer.material, background_fraction=0))
                      for layer in layers]
        old, _ = EELSModel.spectrum_probabilities(old_layers, energy, .5)
        without_background.append(old)
        thickness.append(info['thickness_nm'])
        optical.append(info['t_over_lambda'])
        zlp.append(info['zero_loss_fraction'])
        captured.append(info['detector_window_fraction'])
    axis_nm = (np.arange(256)+.5)*160/256-80
    xx, yy = np.meshgrid(axis_nm, axis_nm)
    image_thickness = 2*np.sqrt(np.maximum(0, 50**2-xx**2-yy**2))
    directory.mkdir(parents=True, exist_ok=True)
    material = EELSModel.EELSMaterial(edges=(), core_fraction=0, background_fraction=1)
    kernel_energy = np.arange(0, 2000.5, .5)
    continuum_density = EELSModel.single_event_kernel(material, kernel_energy)/.5
    np.savez_compressed(directory/'sphere_eels.npz', energy_eV=energy, radius_nm=radii,
        thickness_nm=thickness, t_over_lambda=optical, zero_loss_fraction=zlp,
        detector_window_fraction=captured, spectra_per_incident_electron=spectra,
        ideal_haadf=image_thickness/20, axis_nm=axis_nm,
        spectra_without_background=without_background,
        continuum_energy_eV=kernel_energy, continuum_density_per_eV=continuum_density,
        background_exponent=material.background_exponent,
        background_transition_eV=material.background_transition_eV)
    with (directory/'sphere_eels.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['radius_nm','thickness_nm','t_over_lambda','zero_loss_fraction','detector_window_fraction'])
        writer.writerows(zip(radii, thickness, optical, zlp, captured))
    for row in zip(radii, thickness, optical, zlp):
        print('r={:.0f} nm, t={:.3f} nm, t/lambda={:.4f}, ZLP fraction={:.4f}'.format(*row))


def render(directory):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    with np.load(directory/'sphere_eels.npz') as data:
        energy = data['energy_eV']
        radii = data['radius_nm']
        spectra = data['spectra_per_incident_electron']
        thickness = data['thickness_nm']
        zlp = data['zero_loss_fraction']
        image = data['ideal_haadf']
        old_spectra = data['spectra_without_background']
        continuum_energy = data['continuum_energy_eV']
        continuum_density = data['continuum_density_per_eV']
        exponent = float(data['background_exponent'])
        transition = float(data['background_transition_eV'])
    fig = Figure(figsize=(12, 8), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2)
    colors = matplotlib.colormaps["viridis"](np.linspace(.08, .92, len(radii)))
    axes[0,0].imshow(image, cmap='gray', origin='lower', extent=(-80,80,-80,80))
    for radius, color in zip(radii, colors):
        axes[0,0].plot(radius, 0, 'o', color=color, markeredgecolor='white', markersize=7)
    axes[0,0].set(title='Ideal HAADF: radius 50 nm sphere', xlabel='x (nm)', ylabel='y (nm)')
    rr = np.linspace(0, 60, 601)
    tt = 2*np.sqrt(np.maximum(0, 50**2-rr**2))
    axes[0,1].plot(rr, tt, color='tab:blue', label='Beam-path thickness')
    axes[0,1].scatter(radii, thickness, c=colors)
    axes[0,1].set(xlabel='Distance from sphere center (nm)', ylabel='Thickness (nm)', ylim=(-3,105))
    twin = axes[0,1].twinx()
    twin.plot(rr, np.exp(-tt/100), color='tab:orange', label='Zero-loss fraction')
    twin.scatter(radii, zlp, c=colors)
    twin.set(ylabel='Zero-loss fraction', ylim=(0,1.05))
    axes[0,1].set_title('Geometry drives t/lambda (lambda = 100 nm)')
    axes[0,1].legend(loc='center left')
    twin.legend(loc='center right')
    for radius, spectrum, color in zip(radii, spectra, colors):
        label=f'r = {radius:g} nm' if radius<50 else 'vacuum'
        for axis in axes[1]:
            axis.semilogy(energy, np.maximum(spectrum, 1e-12), color=color, label=label)
    axes[1,0].set(title='Zero-loss and thickness-dependent plural loss', xlim=(-5,120), ylim=(1e-6,1),
        xlabel='Energy loss (eV)', ylabel='Probability per 0.5 eV channel')
    axes[1,1].set(title='Synthetic core-loss spectrum (same incident dose)', xlim=(40,1100), ylim=(1e-7,.1),
        xlabel='Energy loss (eV)', ylabel='Probability per 0.5 eV channel')
    for axis in axes[1]:
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    fig.suptitle('uSim EELS thickness verification - deterministic synthetic model')
    fig.savefig(directory/'sphere_eels_thickness.png', dpi=160)
    print(directory/'sphere_eels_thickness.png')

    comparison = Figure(figsize=(12, 8), layout='constrained')
    FigureCanvasAgg(comparison)
    panels = comparison.subplots(2, 2)
    positive = continuum_energy > 0
    panels[0,0].loglog(continuum_energy[positive], continuum_density[positive], label='Single-event continuum')
    tail = continuum_energy >= transition
    anchor = np.flatnonzero(tail)[0]
    panels[0,0].loglog(continuum_energy[tail], continuum_density[anchor]*(continuum_energy[tail]/transition)**(-exponent),
                       '--', label=f'E^-{exponent:g} reference')
    panels[0,0].axvline(transition, color='gray', alpha=.4)
    panels[0,0].set(title='Smooth low-loss turnover; exact power-law tail', xlabel='Energy loss (eV)', ylabel='Probability density (1/eV)')
    window = (energy >= 300) & (energy < 700)
    for index in (0, 2, 3):
        label = f't={thickness[index]:.1f} nm'
        panels[0,1].semilogy(energy, np.maximum(spectra[index], 1e-12), color=colors[index], label=label+' + continuum')
        panels[0,1].semilogy(energy, np.maximum(old_spectra[index], 1e-12), '--', color=colors[index], label=label+' previous')
        panels[1,0].plot(energy[window], (spectra[index]-old_spectra[index])[window], color=colors[index], label=label)
    panels[0,1].set(title='Same sphere, dose and edges: background on/off', xlim=(200, 1100), ylim=(1e-8, 1e-3),
                     xlabel='Energy loss (eV)', ylabel='Probability per 0.5 eV channel')
    panels[1,0].set(title='Additional continuous pre-edge signal', xlim=(300, 700),
                     xlabel='Energy loss (eV)', ylabel='Probability difference per channel')
    panels[1,1].plot(radii, spectra[:, window].sum(axis=1), 'o-', label='With continuum')
    panels[1,1].plot(radii, old_spectra[:, window].sum(axis=1), 'o--', label='Previous model')
    panels[1,1].set(title='Center -> edge -> vacuum, same incident dose', xlabel='Distance from sphere center (nm)',
                     ylabel='Integrated probability, 300-700 eV')
    for panel in panels.flat:
        panel.legend(fontsize=8)
        panel.grid(alpha=.2)
    comparison.suptitle('Thickness-dependent power-law continuum (synthetic model)')
    comparison.savefig(directory/'sphere_eels_powerlaw_background.png', dpi=160)
    print(directory/'sphere_eels_powerlaw_background.png')


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--data-only', action='store_true')
    modes.add_argument('--render-only', action='store_true')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not args.render_only:
        generate(args.output)
    if not args.data_only:
        render(args.output)


if __name__ == '__main__':
    main()
