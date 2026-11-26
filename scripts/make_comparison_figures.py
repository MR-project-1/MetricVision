"""Comparison figures for the report and README: learned vs. classical front ends, and dynamic-scene handling.

Usage:
    uv run python scripts/make_comparison_figures.py --out docs/comparison/figures \
        --kitti ~/datasets/kitti --euroc ~/datasets/euroc/MH_01_easy

Writes (PDF + PNG):
  matches_<tag>          one image pair matched by every front end (inliers green, outliers red)
  frontend_metrics       drift, per-frame accuracy, inliers and runtime per front end (from results/)
  dynamic_kitti_07       KITTI 07 trajectories with/without dynamic handling, and where the gate / mask fire
Networks are loaded one at a time (MASt3R last) to keep host RAM low.
"""

from __future__ import annotations  # Postponed evaluation of type hints.

import argparse  # Command-line arguments.
import copy  # Deep copies before alignment.
import csv  # Per-frame statistics.
import gc  # Free models between front ends.
import json  # Summaries.
import time  # Timing.
from pathlib import Path  # Paths.
from typing import Any  # Loose typing.

import matplotlib  # Plotting backend selection.

matplotlib.use("Agg")  # Headless rendering.
import cv2  # noqa: E402  # Essential-matrix RANSAC.
import matplotlib.pyplot as plt  # noqa: E402  # Plotting.
import numpy as np  # noqa: E402  # Arrays.
from evo.core import sync  # noqa: E402  # Timestamp association.
from evo.core.trajectory import PoseTrajectory3D  # noqa: E402  # evo trajectory type.
from evo.tools import file_interface  # noqa: E402  # TUM reader.

from metricvision.config import ExperimentConfig  # noqa: E402  # Config parsing.
from metricvision.core.types import ImageView  # noqa: E402  # Matcher input.
from metricvision.datasets import DATASETS  # noqa: E402  # Loaders.
from metricvision.evaluation.kitti_metrics import kitti_segment_errors, load_kitti_poses  # noqa: E402  # Devkit metric.
from metricvision.frontend import MATCHERS  # noqa: E402  # Front ends.

FRONTENDS = [  # (matcher, results folder, label, colour).
    ("orb", "dense_orb_lm", "ORB", "#1f77b4"),
    ("sift", "dense_sift_lm", "SIFT", "#17becf"),
    ("superpoint_lightglue", "dense_superpoint_lg_lm", "SuperPoint+LightGlue", "#9467bd"),
    ("aliked_lightglue", "dense_aliked_lg_lm", "ALIKED+LightGlue", "#e377c2"),
    ("mast3r", "dense_mast3r_lm", "MASt3R", "#8c564b"),
]
SPARSE_BEST = ("sparse_fused_lm_ba5", "Sparse FAST/ORB/KLT + BA (ours)", "#2ca02c")  # Reference pipeline.
DYNAMIC_RUNS = [  # KITTI 07 dynamic-handling variants: (folder, label, colour).
    ("sparse_fused_lm_ba5", "sparse + BA (no dynamic handling)", "#ff7f0e"),
    ("sparse_fused_lm_ba5_gate", "+ geometric gate", "#2ca02c"),
    ("sparse_fused_lm_ba5_mask", "+ semantic mask", "#1f77b4"),
    ("sparse_fused_lm_ba5_gate_mask", "+ gate + mask", "#000080"),
]
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})  # Report-friendly defaults.


def _save(fig: Any, out: Path, name: str) -> None:
    """Save a figure as PDF and PNG.

    Args:
        fig: Matplotlib figure.
        out: Output folder.
        name: File stem.
    """
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")  # Vector.
    fig.savefig(out / f"{name}.png", dpi=170, bbox_inches="tight")  # Raster.
    plt.close(fig)  # Free memory.
    print(f"wrote {out / name}.png")  # Progress.


def _dataset(config_path: str, root: str, sequence: str | None = None) -> tuple[ExperimentConfig, Any]:
    """Parse a config and build its dataset.

    Args:
        config_path: Experiment YAML.
        root: Dataset root.
        sequence: Optional sequence override (KITTI).

    Returns:
        (config, dataset).
    """
    config = ExperimentConfig.from_yaml(config_path).with_overrides(dataset_root=root, sequence=sequence)  # Settings.
    return config, DATASETS.create(config.dataset.name, **config.dataset.params)  # Loader.


def matches_figure(dataset: Any, first: int, second: int, out: Path) -> None:
    """Match one image pair with every front end and draw the flow vectors.

    Args:
        dataset: Loader.
        first: Reference frame index.
        second: Query frame index.
        out: Output folder.
    """
    import torch  # Only to free GPU memory between networks.

    image_a, image_b = dataset.load_frame(first).left, dataset.load_frame(second).left  # The pair.
    camera_matrix = dataset.camera.intrinsic_matrix  # Intrinsics.
    rows = []  # (label, colour, points_a, points_b, inlier mask, ms).
    for name, _, label, colour in FRONTENDS:  # One network at a time.
        try:
            matcher = MATCHERS.create(name, max_keypoints=2000)  # Same budget as the runs.
        except (ImportError, OSError) as error:  # Optional dependency missing.
            print(f"skipping {name}: {error}")
            continue
        matcher.match(ImageView(-1, image_a), ImageView(-2, image_b))  # Warm-up (lazy init, CUDA kernels).
        times = []  # Timings.
        for k in range(3):  # Fresh frame ids so cached features are not reused.
            start = time.perf_counter()  # Timer.
            corr = matcher.match(ImageView(10 * k, image_a), ImageView(10 * k + 1, image_b))  # Match.
            matcher.synchronize()  # Honest GPU timing.
            times.append((time.perf_counter() - start) * 1000.0)  # Milliseconds.
        inliers = np.zeros(len(corr), dtype=bool)  # Essential-matrix RANSAC inliers (1 px).
        if len(corr) >= 5:  # Five-point minimum.
            _, mask = cv2.findEssentialMat(corr.points_a.astype(np.float64), corr.points_b.astype(np.float64), camera_matrix, cv2.RANSAC, 0.999, 1.0)
            inliers = mask.ravel().astype(bool) if mask is not None else inliers
        rows.append((label, colour, corr.points_a, corr.points_b, inliers, float(np.median(times))))  # Keep.
        del matcher  # Free the network...
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # ...and its GPU memory.
    height, width = image_b.shape  # Panel size.
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, len(rows) * 12 * height / width + 0.4 * len(rows)))  # One row per front end.
    for axis, (label, _, pa, pb, inl, ms) in zip(np.atleast_1d(axes), rows, strict=True):  # Draw.
        axis.imshow(image_b, cmap="gray")  # Query image.
        step = max(1, len(pa) // 400)  # Readable number of vectors.
        for mask, colour in ((~inl, "#d62728"), (inl, "#2ca02c")):  # Outliers first, inliers on top.
            idx = np.flatnonzero(mask)[::step]  # Subset.
            axis.plot(np.vstack([pa[idx, 0], pb[idx, 0]]), np.vstack([pa[idx, 1], pb[idx, 1]]), "-", color=colour, lw=0.8)  # Flow.
            axis.scatter(pb[idx, 0], pb[idx, 1], s=3, color=colour)  # End points.
        ratio = inl.mean() if len(inl) else float("nan")  # Inlier ratio.
        axis.set_title(f"{label}: {len(pa)} matches, {100 * ratio:.0f}% inliers (essential RANSAC, 1 px), {ms:.0f} ms/pair", fontsize=10)  # Summary.
        axis.axis("off")  # No ticks.
    fig.suptitle(
        f"{dataset.tag}: frame {first} -> {second}, same pair for every front end (native resolution, 2000-keypoint budget)", y=1.0
    )  # Headline.
    fig.tight_layout()  # Compact.
    _save(fig, out, f"matches_{dataset.tag}")  # Write.


def frontend_metrics_figure(results: Path, kitti_root: Path, out: Path) -> None:
    """Drift, per-frame accuracy, inliers and runtime per front end (dense pipeline), plus the sparse pipeline.

    Args:
        results: Results root.
        kitti_root: KITTI root (for ground-truth poses).
        out: Output folder.
    """
    gt = load_kitti_poses(kitti_root / "poses" / "00.txt")  # KITTI 00 ground truth.
    entries = [(f, lab, col) for _, f, lab, col in FRONTENDS] + [SPARSE_BEST]  # Runs to show.
    rows = []  # Collected metrics.
    for folder, label, colour in entries:  # Each run.
        kitti, euroc = results / "kitti_00" / folder, results / "euroc_MH_01_easy" / folder  # Result folders.
        if not (kitti / "summary.json").is_file():  # Not run yet.
            continue
        s = json.loads((kitti / "summary.json").read_text())  # KITTI summary.
        devkit = kitti_segment_errors(load_kitti_poses(kitti / "trajectory.kitti"), gt)  # Drift.
        euroc_ate = (
            json.loads((euroc / "summary.json").read_text())["errors"]["ate_rmse_m"] if (euroc / "summary.json").is_file() else np.nan
        )  # EuRoC.
        rows.append(
            (
                label,
                colour,
                devkit.translation_percent,
                devkit.rotation_deg_per_100m,
                100 * s["errors"]["rpe_trans_rmse_m"],
                s["mean_pnp_inliers"],
                s["mean_total_ms"],
                100 * euroc_ate,
            )
        )  # Metrics.
    panels = [  # (column, title, unit, number format).
        (2, "KITTI 00 drift t_rel", "%", "%.2f"),
        (3, "KITTI 00 drift r_rel", "deg/100 m", "%.2f"),
        (4, "KITTI 00 per-frame error (RPE)", "cm", "%.2f"),
        (5, "PnP inliers per frame", "count", "%.0f"),
        (6, "Time per frame", "ms", "%.0f"),
        (7, "EuRoC MH_01 ATE", "cm", "%.1f"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 6.6))  # Six panels.
    labels = [r[0] for r in rows]  # Bar labels.
    colours = [r[1] for r in rows]  # Bar colours.
    for axis, (col, title, unit, fmt) in zip(axes.ravel(), panels, strict=True):  # Fill panels.
        values = [r[col] for r in rows]  # Values.
        bars = axis.barh(labels, values, color=colours)  # Bars.
        axis.bar_label(bars, fmt=fmt, fontsize=8, padding=2)  # Numbers.
        axis.invert_yaxis()  # First entry on top.
        axis.set_title(f"{title} [{unit}]", fontsize=10)  # Title.
        axis.set_xlim(0, max(v for v in values if np.isfinite(v)) * 1.22)  # Room for labels.
        if col not in (2, 5):  # Labels only on the left column.
            axis.set_yticklabels([])
    fig.suptitle(
        "Front ends under one fixed back end (dense pipeline: keyframe matching + SGBM depth + PnP + our LM); green = our sparse pipeline", y=1.01
    )  # Headline.
    fig.tight_layout()  # Compact.
    _save(fig, out, "frontend_metrics")  # Write.


def dynamic_figure(dataset: Any, results: Path, out: Path) -> None:
    """KITTI 07 trajectories with/without dynamic handling, and the per-frame rejections of gate and mask.

    Args:
        dataset: KITTI 07 loader.
        results: Results root.
        out: Output folder.
    """
    reference = dataset.ground_truth()  # Ground truth.
    gt = PoseTrajectory3D(poses_se3=list(reference.poses), timestamps=np.asarray(reference.timestamps))  # evo form.
    gt_kitti = load_kitti_poses(Path(dataset._root) / "poses" / "07.txt") if hasattr(dataset, "_root") else None  # Devkit GT (optional).
    fig, (ax_traj, ax_rej) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1, 1.25]})  # Two panels.
    xyz = gt.positions_xyz  # GT positions.
    ax_traj.plot(xyz[:, 0], xyz[:, 2], "k--", lw=1.6, label="ground truth")  # GT.
    for folder, label, colour in DYNAMIC_RUNS:  # Variants.
        run = results / dataset.tag / folder  # Folder.
        if not (run / "trajectory.tum").is_file():  # Not run.
            continue
        ref, est = sync.associate_trajectories(gt, file_interface.read_tum_trajectory_file(str(run / "trajectory.tum")), max_diff=0.01)  # Associate.
        aligned = copy.deepcopy(est)  # Keep original.
        aligned.align(ref, correct_scale=False)  # SE(3).
        ate = float(np.sqrt(np.mean(np.sum((aligned.positions_xyz - ref.positions_xyz) ** 2, axis=1))))  # ATE RMSE.
        extra = ""  # Devkit drift.
        if gt_kitti is not None:
            devkit = kitti_segment_errors(load_kitti_poses(run / "trajectory.kitti"), gt_kitti)
            extra = f", t_rel {devkit.translation_percent:.2f}%"
        width = 4.0 if folder == DYNAMIC_RUNS[0][0] else 1.3  # Baseline as a thick underlay so the variants on top stay visible.
        ax_traj.plot(
            aligned.positions_xyz[:, 0],
            aligned.positions_xyz[:, 2],
            color=colour,
            lw=width,
            alpha=0.55 if width > 2 else 1.0,
            label=f"{label} (ATE {ate:.2f} m{extra})",
        )  # Estimate.
        with (run / "frame_stats.csv").open() as handle:  # Per-frame rejections.
            stats = list(csv.DictReader(handle))
        rejected = np.array([float(r.get("dynamic_rejected") or 0) for r in stats])  # Counts.
        if rejected.any():  # Only variants that reject something.
            kernel = np.ones(15) / 15  # Smoothing.
            ax_rej.plot(np.arange(len(rejected)), np.convolve(rejected, kernel, mode="same"), color=colour, lw=1.2, label=label)  # Curve.
    ax_traj.set_aspect("equal", adjustable="datalim")  # Metric.
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("z [m]")
    ax_traj.set_title(f"{dataset.tag}: top view (SE(3)-aligned)")
    ax_traj.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)  # Below the axes.
    ax_rej.set_xlabel("frame")
    ax_rej.set_ylabel("tracks removed per frame (15-frame mean)")
    ax_rej.set_title("Tracks removed per frame: semantic mask vs. geometric gate")
    ax_rej.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out, f"dynamic_{dataset.tag}")


def main() -> None:
    """Generate all comparison figures."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # CLI.
    parser.add_argument("--out", type=Path, default=Path("docs/comparison/figures"))  # Output folder.
    parser.add_argument("--results", type=Path, default=Path("results"))  # Results root.
    parser.add_argument("--kitti", required=True)  # KITTI root.
    parser.add_argument("--euroc", required=True)  # EuRoC scene root.
    parser.add_argument("--skip-matches", action="store_true")  # Reuse existing match figures (they load every network).
    args = parser.parse_args()  # Parse.
    args.out.mkdir(parents=True, exist_ok=True)  # Output folder.
    kitti_root = Path(args.kitti).expanduser()  # Paths.
    _, kitti00 = _dataset("configs/kitti.yaml", str(kitti_root))  # KITTI 00.
    _, euroc = _dataset("configs/euroc.yaml", str(Path(args.euroc).expanduser()))  # EuRoC.
    frontend_metrics_figure(args.results, kitti_root, args.out)  # From results/.
    if (kitti_root / "sequences" / "07").is_dir():  # Dynamic scene.
        _, kitti07 = _dataset("configs/kitti.yaml", str(kitti_root), "07")
        dynamic_figure(kitti07, args.results, args.out)
    if not args.skip_matches:  # Loads every network in turn.
        matches_figure(kitti00, 300, 303, args.out)
        matches_figure(euroc, 1200, 1206, args.out)


if __name__ == "__main__":  # Script entry.
    main()  # Run.
