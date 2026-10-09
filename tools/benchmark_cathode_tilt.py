"""Measure warmed computation per cathode tilt update, excluding UI/exposure."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from nion.utils import Geometry
from nion.instrumentation import stem_controller
from nion.usim_device import InstrumentDevice, RonchigramCameraSimulator, HAADFFocusModel


def benchmark(size, steps):
    generator = InstrumentDevice.ScanDataGenerator(sample_index=8)
    manager = InstrumentDevice.ValueManager()
    instrument = InstrumentDevice.Instrument('usim_stem_controller', manager, InstrumentDevice.AxisManager(), generator)
    sample = generator.sample
    stage, fov = sample.initial_view
    instrument.stage_position_m = stage
    instrument.SetVal('C10', 1000e-9)
    camera = RonchigramCameraSimulator.RonchigramCameraSimulator(instrument, Geometry.IntSize(size, size), 10, instrument.stage_size_nm)
    camera.noise.enabled = False
    area = Geometry.IntRect(origin=Geometry.IntPoint(), size=Geometry.IntSize(size, size))
    context = stem_controller.ScanContext(Geometry.IntSize(size, size), Geometry.FloatPoint(), fov, 0.)
    results = []
    try:
        for step in range(steps+1):
            tilt = Geometry.FloatPoint(x=np.radians(.1*(step+1)), y=np.radians(.03*(step+1)))
            manager.set_value_2d('stage_tilt_rad', tilt)
            start = perf_counter()
            sample.set_stage_tilt(tilt)
            geometry = perf_counter()
            planes = sample.generate_depth_planes(stage, Geometry.FloatSize(fov, fov), Geometry.FloatPoint(), Geometry.FloatPoint(), Geometry.IntSize(size, size), 2.)
            depth = perf_counter()
            HAADFFocusModel.apply_depth_planes_defocus(planes, defocus_m=1000e-9, best_focus_m=0.,
                convergence_angle_rad=instrument.GetVal('ConvergenceAngle'), pixel_size_y_nm=fov/size, pixel_size_x_nm=fov/size)
            scan = perf_counter()
            frame = camera.get_frame_data(area, Geometry.IntSize(1, 1), .01, context, Geometry.FloatPoint(.5, .5))
            end = perf_counter()
            if step:
                results.append(dict(geometry_ms=1000*(geometry-start), depth_ms=1000*(depth-geometry),
                    haadf_render_ms=1000*(scan-depth), ronchigram_ms=1000*(end-scan),
                    planes=len(planes), ronchigram_backend=frame.metadata['kikuchi_simulation'].get('compute_backend')))
        return {'size': size, 'steps': steps, 'timings': results,
            'median_ms': {name: float(np.median([r[name] for r in results])) for name in results[0] if name.endswith('_ms')}}
    finally:
        camera.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = benchmark(args.size, args.steps)
    serialized = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(serialized, encoding='utf-8')
    print(serialized)
