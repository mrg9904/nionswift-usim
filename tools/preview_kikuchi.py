"""Generate actual Ronchigram frames in nionswift-dev; render with Matplotlib."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

OUTPUT = Path(__file__).resolve().parent / 'kikuchi_results'


def generate():
    from types import SimpleNamespace
    from nion.instrumentation import stem_controller
    from nion.usim_device import InstrumentDevice, RonchigramCameraSimulator, SampleSimulator
    from nion.utils import Event, Geometry

    class Scope(SimpleNamespace):
        @property
        def stage_position_m(self):
            return self.value_manager.stage_position_m

    manager = InstrumentDevice.ValueManager()
    sample = SampleSimulator.SphericalParticleSample(1000)
    scope = Scope(value_manager=manager, scan_data_generator=SimpleNamespace(sample=sample),
        scan_controller=SimpleNamespace(scan_device=SimpleNamespace()), probe_state='parked',
        probe_state_changed_event=Event.Event(), GetVal=manager.get_value, GetVal2D=manager.get_value_2d,
        max_defocus=5000e-9, stage_size_nm=1000, defocus_m=0.)
    shape = Geometry.IntSize(512, 512)
    camera = RonchigramCameraSimulator.RonchigramCameraSimulator(scope, shape, 10, 1000)
    camera.noise.enabled = False
    context = stem_controller.ScanContext(Geometry.IntSize(64, 64), Geometry.FloatPoint(), 200., 0.)
    area = Geometry.IntRect(origin=Geometry.IntPoint(), size=shape)
    cases = [('center, TX=TY=0', 0, 0, .5, .5), ('center, TX=+1 deg', 1, 0, .5, .5),
             ('center, TY=+1 deg', 0, 1, .5, .5), ('center, TX=TY=+1 deg', 1, 1, .5, .5),
             ('edge, r=49 nm', 0, 0, .5, .745), ('vacuum, r=60 nm', 0, 0, .5, .8)]
    cases += [('defocus=500 nm, aperture OUT', 0, 0, .5, .5),
              ('defocus=500 nm, aperture IN', 0, 0, .5, .5),
              ('defocus=1000 nm, beam centre in vacuum', 0, 0, .5, .8)]
    frames, noisy_frames, infos, titles = [], [], [], []
    try:
        for title, tx, ty, y, x in cases:
            manager.set_value('C10', 1000e-9 if 'defocus=1000' in title else 500e-9 if 'defocus=' in title else 0.)
            manager.set_value('S_VOA', 1 if 'aperture IN' in title else 0)
            manager.set_value_2d('stage_tilt_rad', Geometry.FloatPoint(x=np.deg2rad(tx), y=np.deg2rad(ty)))
            start = time.perf_counter()
            frame = camera.get_frame_data(area, Geometry.IntSize(1, 1), .1, context, Geometry.FloatPoint(y=y, x=x))
            info = frame.metadata['kikuchi_simulation']
            info['frame_seconds'] = time.perf_counter()-start
            frames.append(frame.data)
            camera.noise.enabled = True
            noisy_frames.append(camera.get_frame_data(area, Geometry.IntSize(1, 1), .1, context,
                Geometry.FloatPoint(y=y, x=x)).data)
            camera.noise.enabled = False
            infos.append(info)
            titles.append(title)
        cal = frame.dimensional_calibrations
        from unittest.mock import patch
        from nion.usim_device import KikuchiModel
        manager.set_value('C10', 1000e-9)
        manager.set_value('S_VOA', 0)
        manager.set_value_2d('stage_tilt_rad', Geometry.FloatPoint())
        def capture():
            return camera.get_frame_data(area, Geometry.IntSize(1, 1), .1, context, Geometry.FloatPoint(.5, .5)).data.copy()
        with patch.object(KikuchiModel, 'thickness_contrast', return_value=0):
            real_and_diffuse = capture()
        camera._needs_recalculation = True
        composite = capture()
        camera.noise.enabled = True
        composite_noisy = capture()
        profile_axis_rad = np.linspace(-.006, .006, 2049)
        pixel_angle = profile_axis_rad[1]-profile_axis_rad[0]
        isolated_line = np.tile(np.exp(-.5*(profile_axis_rad/.00035)**2), (4, 1))
        profile_thicknesses = [20., 50., 100., 200.]
        profiles = []
        for thickness in profile_thicknesses:
            real = np.full(isolated_line.shape, 100*KikuchiModel.transmission(thickness))
            result, _ = KikuchiModel.compose_contrast(real, 100, isolated_line, (pixel_angle, pixel_angle))
            background, _ = KikuchiModel.compose_contrast(real, 100, isolated_line*0, (pixel_angle, pixel_angle))
            profiles.append((result-background)[0])
        OUTPUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUTPUT/'ronchigram_kikuchi.npz', frames=frames, noisy_frames=noisy_frames, titles=titles,
            real_and_diffuse=real_and_diffuse, diffraction_component=composite-real_and_diffuse,
            composite_noisy=composite_noisy, profile_axis_mrad=profile_axis_rad*1000,
            profile_thicknesses_nm=profile_thicknesses, profiles=profiles,
            x_mrad=(cal[1].offset + np.arange(shape.width)*cal[1].scale)*1000,
            y_mrad=(cal[0].offset + np.arange(shape.height)*cal[0].scale)*1000)
        (OUTPUT/'ronchigram_kikuchi.json').write_text(json.dumps(infos, indent=2), encoding='utf-8')
        print(json.dumps({'output': str(OUTPUT), 'frame_seconds': [i['frame_seconds'] for i in infos]}))
    finally:
        camera.close()


def render():
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    data = np.load(OUTPUT/'ronchigram_kikuchi.npz')
    infos = json.loads((OUTPUT/'ronchigram_kikuchi.json').read_text())
    figure = Figure(figsize=(12, 12), layout='constrained')
    FigureCanvasAgg(figure)
    extent = [data['x_mrad'][0], data['x_mrad'][-1], data['y_mrad'][-1], data['y_mrad'][0]]
    vmin, vmax = data['frames'].min(), data['frames'].max()
    for index, (frame, title, info) in enumerate(zip(data['frames'], data['titles'], infos)):
        ax = figure.add_subplot(3, 3, index+1)
        ax.imshow(frame, extent=extent, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_title(f"{title}\nt={info['thickness_nm']:.1f} nm, {info['band_count']} plane pairs")
        ax.set_xlabel('x (mrad)')
        ax.set_ylabel('y (mrad)')
    figure.suptitle('NNMTO CIF: [001], 100 kV, 40 mrad convergence\nGeometric line positions; illustrative intensity, common display scale')
    target = OUTPUT/'ronchigram_kikuchi.png'
    figure.savefig(target, dpi=160)
    print(target)
    noise_figure = Figure(figsize=(10, 8), layout='constrained')
    FigureCanvasAgg(noise_figure)
    for row, index in enumerate((0, 5)):
        for column, key in enumerate(('frames', 'noisy_frames')):
            ax = noise_figure.add_subplot(2, 2, row*2+column+1)
            ax.imshow(data[key][index], extent=extent, cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(f"{data['titles'][index]}: {'ideal' if column == 0 else 'electron shot noise'}")
            ax.set_xlabel('x (mrad)')
            ax.set_ylabel('y (mrad)')
    noise_figure.savefig(OUTPUT/'ronchigram_kikuchi_noise.png', dpi=160)
    offaxis_figure = Figure(figsize=(7, 7), layout='constrained')
    FigureCanvasAgg(offaxis_figure)
    ax = offaxis_figure.add_subplot(1, 1, 1)
    ax.imshow(data['noisy_frames'][8], extent=extent, cmap='gray', vmin=0, vmax=vmax)
    ax.plot(0, 0, '+', color='red', markersize=12)
    ax.set_title('Beam centre in vacuum (red +), defocus=1000 nm\nOff-centre illuminated sphere still shows Kikuchi lines')
    ax.set_xlabel('x (mrad)')
    ax.set_ylabel('y (mrad)')
    offaxis_figure.savefig(OUTPUT/'ronchigram_offaxis_particle.png', dpi=160)
    contrast_figure = Figure(figsize=(15, 9), layout='constrained')
    FigureCanvasAgg(contrast_figure)
    grid = contrast_figure.add_gridspec(2, 3)
    for index, (key, label) in enumerate((('real_and_diffuse', 'Real-space transmission + diffuse background'),
        ('diffraction_component', 'Kikuchi contribution (signed contrast)'),
        ('composite_noisy', 'Combined Ronchigram + shot noise'))):
        ax = contrast_figure.add_subplot(grid[0, index])
        if index == 1:
            limit = abs(data[key]).max()
            ax.imshow(data[key], extent=extent, cmap='coolwarm', vmin=-limit, vmax=limit)
        else:
            ax.imshow(data[key], extent=extent, cmap='gray', vmin=0, vmax=data['real_and_diffuse'].max())
        ax.set_title(label)
        ax.set_xlabel('x (mrad)')
        ax.set_ylabel('y (mrad)')
    ax = contrast_figure.add_subplot(grid[1, :])
    for thickness, profile in zip(data['profile_thicknesses_nm'], data['profiles']):
        ax.plot(data['profile_axis_mrad'], profile, label=f'{thickness:g} nm')
    ax.set_xlim(-4, 4)
    ax.set_xlabel('Angular distance from one geometric line (mrad)')
    ax.set_ylabel('Line contribution (vacuum = 100)')
    ax.set_title('Increasing thickness: broader lines, lower peaks; common scale, no peak normalization')
    ax.legend()
    contrast_figure.suptitle('Sphere: [001], 100 kV, defocus 1000 nm, aperture OUT\nPhenomenological real-space / diffraction / diffuse mixture')
    contrast_figure.savefig(OUTPUT/'ronchigram_contrast_thickness.png', dpi=160)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-only', action='store_true')
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    if not args.render_only:
        generate()
    if not args.data_only:
        render()
