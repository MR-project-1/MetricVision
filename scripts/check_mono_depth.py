"""Compare a monocular metric depth network against SGBM stereo depth (scale, accuracy, speed, VRAM).

Usage:
    uv run python scripts/check_mono_depth.py --config configs/kitti.yaml --dataset-root ~/datasets/kitti --frames 20

Stereo depth is only a reference here (it is itself noisy far away), so the comparison is
restricted to pixels where SGBM is valid and closer than ``--max-ref-depth``.
"""

from __future__ import annotations  # Postponed evaluation of type hints.

import argparse  # Command-line arguments.
import time  # Timing.

import numpy as np  # Statistics.

from metricvision.config import ExperimentConfig  # Config parsing.
from metricvision.datasets import DATASETS  # Loaders.
from metricvision.depth import DEPTH_ESTIMATORS  # Estimators.


def main() -> None:
    """Run the comparison and print a summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # CLI.
    parser.add_argument("--config", required=True)  # Dataset config.
    parser.add_argument("--dataset-root", required=True)  # Data location.
    parser.add_argument("--frames", type=int, default=20)  # Frames sampled across the sequence.
    parser.add_argument("--max-ref-depth", type=float, default=20.0)  # Trust SGBM only up to here.
    args = parser.parse_args()  # Parse.
    config = ExperimentConfig.from_yaml(args.config).with_overrides(dataset_root=args.dataset_root)  # Settings.
    dataset = DATASETS.create(config.dataset.name, **config.dataset.params)  # Loader.
    network = DEPTH_ESTIMATORS.create(config.prior_depth.name, **config.prior_depth.params)  # Mono depth.
    stereo = DEPTH_ESTIMATORS.create(config.depth.name, **config.depth.params)  # SGBM reference.
    ratios, abs_rel, times = [], [], []  # Per-frame statistics.
    for index in np.linspace(0, len(dataset) - 1, args.frames).astype(int):  # Spread over the sequence.
        frame = dataset.load_frame(int(index))  # Stereo pair.
        start = time.perf_counter()  # Time the network.
        mono = network.compute(frame.left, frame.right, dataset.camera)  # Network depth.
        times.append((time.perf_counter() - start) * 1000.0)  # Milliseconds.
        ref = stereo.compute(frame.left, frame.right, dataset.camera)  # Stereo depth.
        valid = np.isfinite(mono) & np.isfinite(ref) & (ref < args.max_ref_depth)  # Comparable pixels.
        if valid.sum() < 100:  # Too few.
            continue
        ratios.append(float(np.median(ref[valid] / mono[valid])))  # Stereo / network scale.
        abs_rel.append(float(np.median(np.abs(mono[valid] - ref[valid]) / ref[valid])))  # Median relative error (raw scale).
    print(f"{dataset.tag}: {len(ratios)} frames, model {config.prior_depth.params}")  # Header.
    print(f"  scale ratio stereo/network: median {np.median(ratios):.3f}, range [{min(ratios):.3f}, {max(ratios):.3f}]")  # Scale.
    print(f"  median |mono - stereo| / stereo (raw, no alignment): {np.median(abs_rel):.3f}")  # Accuracy.
    print(f"  network time: median {np.median(times[1:]):.1f} ms (first call {times[0]:.0f} ms incl. warm-up)")  # Speed.
    print(f"  peak GPU memory (PyTorch allocator): {getattr(network, 'peak_memory_mb', 0.0):.0f} MiB")  # VRAM.


if __name__ == "__main__":  # Script entry.
    main()  # Run.
