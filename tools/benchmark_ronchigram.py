"""Measure moving-probe frame latency and compare a saved pre-optimization image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
import numpy as np
from nion.usim_device import SimulationSettings
from nion.utils import Geometry
from nionswift_plugin.usim.test.KikuchiModel_test import TestKikuchiModel


def benchmark(backend, binning, frames):
    SimulationSettings.RONCHIGRAM_BACKEND = backend
    fixture = TestKikuchiModel()
    fixture.setUp()
    manager, camera, context, area = fixture.make_camera(2048)
    manager.set_value("C10Control", 1000e-9)
    camera.noise.enabled = True
    bins = Geometry.IntSize(binning, binning)
    try:
        start = time.perf_counter()
        first = camera.get_frame_data(area, bins, .1, context, Geometry.FloatPoint(.5, .5))
        cold = time.perf_counter()-start
        elapsed = []
        for index in range(frames):
            position = Geometry.FloatPoint(.5, .52+.28*(index % 10)/9)
            start = time.perf_counter()
            camera.get_frame_data(area, bins, .1, context, position)
            elapsed.append(time.perf_counter()-start)
        result = {"requested_backend": backend, "actual_backend": first.metadata["kikuchi_simulation"].get("compute_backend"),
                  "shape": list(first.data.shape), "noise": True, "exposure_s": .1,
                  "cold_start_s": cold, "median_s": statistics.median(elapsed),
                  "p95_s": float(np.percentile(elapsed, 95)), "frames": frames}
        reference = Path("tools/kikuchi_results/performance_reference.npz")
        if binning == 2 and reference.exists():
            camera.noise.enabled = False
            expected = np.load(reference)["frames"]
            observed = np.array([camera.get_frame_data(area, bins, .1, context, Geometry.FloatPoint(.5, x)).data.copy()
                                 for x in [.5, .55, .6, .7, .8]])
            result["reference_max_absolute_error_counts"] = float(np.max(np.abs(expected-observed)))
            result["reference_max_relative_to_vacuum"] = float(np.max(np.abs(expected-observed))/np.max(expected))
        return result
    finally:
        camera.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--backends", nargs="+", default=["cpu", "gpu"])
    args = parser.parse_args()
    results = [benchmark(backend, bins, args.frames) for bins in (2, 1) for backend in args.backends]
    print(json.dumps(results, indent=2))
    path = Path("tools/kikuchi_results/ronchigram_performance.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
