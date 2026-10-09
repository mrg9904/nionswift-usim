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
    from nion.usim_device import EELSModel, SampleSimulator
    from nion.utils import Geometry
    sphere = SampleSimulator.SphericalParticleSample(1000)
    energy = np.arange(-20., 1400., .5)
    radii = np.asarray([0., 20., 40., 49., 55.])
    spectra, thickness, optical, zlp, captured = [], [], [], [], []
    for radius in radii:
        layers = sphere.eels_layers_at(Geometry.FloatPoint(x=radius*1e-9, y=0))
        spectrum, info = EELSModel.spectrum_probabilities(layers, energy, .5)
        spectra.append(spectrum)
        thickness.append(info['thickness_nm'])
        optical.append(info['t_over_lambda'])
        zlp.append(info['zero_loss_fraction'])
        captured.append(info['detector_window_fraction'])
    axis_nm = (np.arange(256)+.5)*160/256-80
    xx, yy = np.meshgrid(axis_nm, axis_nm)
    image_thickness = 2*np.sqrt(np.maximum(0, 50**2-xx**2-yy**2))
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory/'sphere_eels.npz', energy_eV=energy, radius_nm=radii,
        thickness_nm=thickness, t_over_lambda=optical, zero_loss_fraction=zlp,
        detector_window_fraction=captured, spectra_per_incident_electron=spectra,
        ideal_haadf=image_thickness/20, axis_nm=axis_nm)
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
