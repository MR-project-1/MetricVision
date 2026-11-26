"""Watch both approaches run on a bag or dataset, next to the reference, and save the graph as proof.

Frame by frame, the classical (FAST/ORB + KLT) and the learned (SuperPoint + LightGlue) front end run on the same
frames with the same rigid-rig back end. A window shows, live:

* top: reference (wheel /odom for the wheelchair bags, ground truth for KITTI/EuRoC), classical and learned path,
  all anchored at the first frame (same colours as the live graph window and rviz2),
* bottom: each approach's distance to the reference over time.

Buttons: **Save graph** (PNG) and **Save results** (graph + both trajectories + reference + summary.json) into
``results/simulation/compare/<sequence>_<time>/``. Everything is also saved at the end of the sequence.

Usage:
    uv run python scripts/compare_live.py --bag ros_bags/run_01
    uv run python scripts/compare_live.py --config configs/kitti.yaml --dataset-root /data/kitti --sequence 00 --max-frames 1500
    uv run python scripts/compare_live.py --bag ros_bags/small_run_02 --no-show      # no window, just save at the end
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from metricvision.config import ExperimentConfig
from metricvision.core.geometry import invert_pose
from metricvision.core.types import Trajectory
from metricvision.datasets import DATASETS
from metricvision.datasets.rosbag_stereo import interpolate_poses
from metricvision.evaluation.writers import write_tum
from metricvision.pipeline import build_odometry

APPROACHES = {"classical": ("fast_orb_klt", "#1f7ae0"), "learned": ("superpoint_lightglue", "#8e44ad")}
REFERENCE_COLOUR = "#1a9b55"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/wheelchair.yaml"))
    parser.add_argument("--bag", help="Wheelchair bag folder (shortcut for --dataset-root with the wheelchair config).")
    parser.add_argument("--dataset-root")
    parser.add_argument("--sequence")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--window", type=int, default=5, help="Sliding-window BA size (0 = off).")
    parser.add_argument("--out", type=Path, default=Path("results/simulation/compare"))
    parser.add_argument("--no-show", action="store_true", help="No window; save at the end only.")
    parser.add_argument("--redraw-every", type=int, default=5, help="Redraw the window every N frames.")
    args = parser.parse_args()

    import matplotlib

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    base = ExperimentConfig.from_yaml(args.config).with_overrides(
        dataset_root=args.bag or args.dataset_root, sequence=args.sequence, pipeline="sparse", window=args.window
    )
    dataset = DATASETS.create(base.dataset.name, **base.dataset.params)
    pipelines = {name: build_odometry(base.with_overrides(stereo_frontend=frontend), dataset.camera) for name, (frontend, _) in APPROACHES.items()}
    for odometry in pipelines.values():
        odometry.reset()
    reference = dataset.ground_truth()
    n = min(len(dataset), args.max_frames or len(dataset))
    folder = args.out / f"{dataset.tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    body = dataset.body_from_camera
    estimates: dict[str, Trajectory] = {name: Trajectory() for name in APPROACHES}
    ms: dict[str, list[float]] = {name: [] for name in APPROACHES}
    failures = dict.fromkeys(APPROACHES, 0)

    ref_t = np.asarray(reference.timestamps) if reference is not None else np.zeros(0)
    ref_p = list(reference.poses) if reference is not None else []

    def reference_at(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(ref_t) < 2 or len(times) == 0:
            return np.zeros((0, 4, 4)), np.zeros(0, dtype=bool)
        return interpolate_poses(ref_t, ref_p, times)

    def anchored(trajectory: Trajectory) -> np.ndarray:
        if not len(trajectory):
            return np.zeros((0, 4, 4))
        inv = invert_pose(trajectory.poses[0])
        return np.array([inv @ p for p in trajectory.poses])

    def plane() -> tuple[int, int]:
        poses = np.array(ref_p) if ref_p else anchored(estimates["classical"])
        spread = np.ptp(poses[:, :3, 3], axis=0) if len(poses) else np.ones(3)
        return (0, 1) if spread[2] <= min(spread[0], spread[1]) else (0, 2)

    fig = plt.figure(f"MetricVision simulation: {dataset.tag}", figsize=(8.5, 8.5))
    ax_map = fig.add_axes((0.09, 0.40, 0.88, 0.55))
    ax_err = fig.add_axes((0.09, 0.13, 0.88, 0.19))
    status = fig.text(0.70, 0.04, "", fontsize=8, va="center")
    state: dict[str, Any] = {"frame": 0}

    def draw() -> dict[str, dict[str, float]]:
        a, b = plane()
        ax_map.clear()
        ax_err.clear()
        stats: dict[str, dict[str, float]] = {}
        first = estimates["classical"]
        ref_line = None
        if len(first):
            ref_i, inside = reference_at(np.asarray(first.timestamps))
            if len(ref_i):
                ref_line = np.array([invert_pose(ref_i[0]) @ p for p in ref_i])
                ax_map.plot(ref_line[:, a, 3], ref_line[:, b, 3], color=REFERENCE_COLOUR, lw=2.6, label="reference")
        for name, (_, colour) in APPROACHES.items():
            est = anchored(estimates[name])
            if not len(est):
                continue
            ax_map.plot(est[:, a, 3], est[:, b, 3], color=colour, lw=1.6, label=f"{name} ({np.mean(ms[name]):.0f} ms/frame)")
            t = np.asarray(estimates[name].timestamps)
            ref_i, inside = reference_at(t)
            if len(ref_i) > 1:
                ref_i = np.array([invert_pose(ref_i[0]) @ p for p in ref_i])
                diff = np.linalg.norm(est[inside][:, :3, 3] - ref_i[:, :3, 3], axis=1)
                ax_err.plot(t[inside] - t[0], diff, color=colour, label=name)
                stats[name] = {"end_diff_m": float(diff[-1]), "mean_diff_m": float(diff.mean()), "tracking_failures": failures[name]}
        ax_map.plot(0, 0, "k+", ms=10)
        ax_map.set_title(f"{dataset.tag}: frame {state['frame']}/{n}, all anchored at the first frame")
        ax_map.set_xlabel("xyz"[a] + " [m]")
        ax_map.set_ylabel("xyz"[b] + " [m]")
        ax_map.axis("equal")
        ax_map.grid(alpha=0.3)
        ax_map.legend(fontsize=8, loc="best")
        ax_err.set_title("Distance to the reference (anchored, no fitting)", fontsize=9)
        ax_err.set_xlabel("time [s]")
        ax_err.set_ylabel("[m]")
        ax_err.grid(alpha=0.3)
        if stats:
            ax_err.legend(fontsize=8, loc="upper left")
        return stats

    def save_graph(_: Any = None) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"compare_frame{state['frame']:05d}.png"
        fig.savefig(path, dpi=150)
        status.set_text(f"saved {path.name}")
        return path

    def save_results(_: Any = None) -> None:
        stats = draw()
        save_graph()
        for name, trajectory in estimates.items():
            write_tum(trajectory, folder / f"{name}.tum")
        if reference is not None:
            write_tum(reference, folder / "reference.tum")
        summary = {
            "sequence": dataset.tag,
            "frames": state["frame"],
            "approaches": {name: {**stats.get(name, {}), "ms_per_frame": float(np.mean(ms[name])) if ms[name] else None} for name in APPROACHES},
        }
        (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        status.set_text(f"saved results to {folder.name}")
        print(f"saved {folder}")

    if not args.no_show:
        buttons = (Button(fig.add_axes((0.09, 0.02, 0.25, 0.055)), "Save graph"), Button(fig.add_axes((0.37, 0.02, 0.30, 0.055)), "Save results"))
        buttons[0].on_clicked(save_graph)
        buttons[1].on_clicked(save_results)
        plt.ion()
        plt.show()

    for index in range(n):
        frame = dataset.load_frame(index)
        for name, odometry in pipelines.items():
            start = time.perf_counter()
            result = odometry.process(frame)
            ms[name].append(1e3 * (time.perf_counter() - start))
            failures[name] += 0 if result.stats.tracking_ok else 1
            estimates[name].append(frame.timestamp, body @ result.world_from_camera @ invert_pose(body))
        state["frame"] = index + 1
        if not args.no_show and (index % args.redraw_every == 0 or index == n - 1):
            if not plt.fignum_exists(fig.number):
                print("window closed: stopping")
                break
            draw()
            plt.pause(0.001)
    save_results()
    if not args.no_show and plt.fignum_exists(fig.number):
        plt.ioff()
        plt.show()  # Keep the window open for a last look / manual saves.


if __name__ == "__main__":
    main()
