"""Run with the Nion Python environment; includes device synchronization."""
import argparse
import time

from nion.usim_device import HAADFFocusModel, Noise, SimulationSettings, STLDepthSample
from nion.utils import Geometry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1028)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    SimulationSettings.STL_SURFACE_BACKEND = args.backend
    sample = STLDepthSample.STLDepthSample(10000)
    noise = Noise.HAADFNoise(seed=42)
    if args.backend == "gpu":
        import cupy as cp
        print("GPU:", cp.cuda.runtime.getDeviceProperties(0)["name"])
        sync = cp.cuda.Stream.null.synchronize
    else:
        sync = lambda: None
    for fov in (100., 80., 101., 81.):
        sync()
        start = time.perf_counter()
        planes = sample.generate_depth_planes(
            Geometry.FloatPoint(), Geometry.FloatSize(fov, fov), Geometry.FloatPoint(),
            Geometry.FloatPoint(), Geometry.IntSize(args.size, args.size),
            SimulationSettings.calculate_depth_slice_thickness_nm(fov, fov))
        sync()
        geometry_ms = (time.perf_counter() - start) * 1000
        if args.backend == "gpu":
            assert getattr(planes, "_on_gpu", False), "GPU path fell back to CPU"
        for focus in (500., 510., 0.):
            start = time.perf_counter()
            image = HAADFFocusModel.apply_depth_planes_defocus(
                planes, defocus_m=focus * 1e-9, best_focus_m=0., convergence_angle_rad=.04,
                pixel_size_y_nm=fov / args.size, pixel_size_x_nm=fov / args.size,
                return_device=args.backend == "gpu")
            sync()
            render_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            image = noise.apply(image, 1., 200e-12)
            if args.backend == "gpu":
                image = image.get()
            sync()
            noise_ms = (time.perf_counter() - start) * 1000
            print(f"FoV={fov:g} planes={len(planes)} focus={focus:g} "
                  f"geometry={geometry_ms:.1f}ms render={render_ms:.1f}ms "
                  f"noise+transfer={noise_ms:.1f}ms mean={image.mean():.5f}", flush=True)


if __name__ == "__main__":
    main()
