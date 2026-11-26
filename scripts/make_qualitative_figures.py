"""Qualitative figures for the report: what the sparse front end sees, depth uncertainty, LM convergence.

Usage:
    uv run python scripts/make_qualitative_figures.py --out docs/figures \
        --config configs/kitti.yaml:/data/kitti:300 --config configs/euroc.yaml:/data/euroc/MH_01_easy:1200

Each ``--config`` is ``<yaml>:<dataset root>:<frame index>``. For every dataset it writes:
``<tag>_stereo_matches`` (FAST/ORB epipolar matches, coloured by depth),
``<tag>_klt_tracks`` (KLT tracks into the next frame: circular-check survivors vs rejects),
and, across datasets, ``depth_uncertainty`` (sigma_Z vs Z) and ``lm_convergence``.
"""

from __future__ import annotations  # Postponed evaluation of type hints.

import argparse  # Command-line arguments.
from pathlib import Path  # Paths.
from typing import Any  # Loose typing for plot handles.

import matplotlib  # Plotting backend selection.

matplotlib.use("Agg")  # Headless rendering.
import cv2  # noqa: E402  # Raw KLT for visualising rejected tracks.
import matplotlib.pyplot as plt  # noqa: E402  # Plotting.
import numpy as np  # noqa: E402  # Arrays.

from metricvision.backend.lm import LMConfig, refine_pose_lm  # noqa: E402  # Rigid-rig LM.
from metricvision.backend.pose import PnPConfig, PnPPoseEstimator  # noqa: E402  # RANSAC initialisation.
from metricvision.config import ExperimentConfig  # noqa: E402  # Config parsing.
from metricvision.core.geometry import se3_exp  # noqa: E402  # Perturbations.
from metricvision.core.stereo import triangulate_rectified, triangulation_covariance  # noqa: E402  # Depth + covariance.
from metricvision.datasets import DATASETS  # noqa: E402  # Loaders.
from metricvision.frontend import STEREO_FRONTENDS  # noqa: E402  # Sparse front end.
from metricvision.frontend.stereo_klt import StereoTracks  # noqa: E402  # Track container.


def _save(fig: Any, out: Path, name: str) -> None:
    """Save a figure as PDF (LaTeX) and PNG (Markdown).

    Args:
        fig: Matplotlib figure.
        out: Output folder.
        name: File stem.
    """
    fig.tight_layout()  # Compact layout.
    fig.savefig(out / f"{name}.pdf")  # Vector.
    fig.savefig(out / f"{name}.png", dpi=150)  # Raster.
    plt.close(fig)  # Free memory.


def _stereo_figure(left: np.ndarray, right: np.ndarray, pairs: StereoTracks, depth: np.ndarray, tag: str, out: Path) -> None:
    """Left/right images stacked with epipolar matches drawn as lines, coloured by depth.

    Args:
        left: Left image.
        right: Right image.
        pairs: Stereo pairs.
        depth: (N,) depths of the pairs.
        tag: Dataset tag.
        out: Output folder.
    """
    height = left.shape[0]  # Offset of the right image in the stack.
    fig, ax = plt.subplots(figsize=(10, 10 * 2 * height / left.shape[1] + 0.6))  # Aspect-preserving.
    ax.imshow(np.vstack([left, right]), cmap="gray")  # Stacked pair.
    cmap = plt.get_cmap("turbo")  # Depth colours.
    finite = np.isfinite(depth)  # Valid depths.
    vmax = float(np.percentile(depth[finite], 95)) if finite.any() else 1.0  # Robust colour range.
    order = np.arange(len(pairs))[:: max(1, len(pairs) // 150)]  # Draw a readable subset of lines.
    for i in order:  # One line per match.
        colour = cmap(min(depth[i] / vmax, 1.0)) if np.isfinite(depth[i]) else (1, 0, 0, 1)  # Depth colour.
        ax.plot([pairs.left[i, 0], pairs.right[i, 0]], [pairs.left[i, 1], pairs.right[i, 1] + height], "-", color=colour, lw=0.6)  # Match line.
    sc = ax.scatter(pairs.left[:, 0], pairs.left[:, 1], c=np.minimum(depth, vmax), cmap=cmap, s=5, vmin=0, vmax=vmax)  # All left features.
    fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01, label="depth [m]")  # Legend.
    ax.set_title(
        f"{tag}: FAST + ORB stereo matches along epipolar lines ({len(pairs)} pairs; every {max(1, len(pairs) // 150)}th line drawn)"
    )  # Title.
    ax.axis("off")  # No ticks.
    _save(fig, out, f"{tag}_stereo_matches")  # Write.


def _klt_figure(prev_left: np.ndarray, cur_left: np.ndarray, pairs: StereoTracks, keep: np.ndarray, tag: str, out: Path) -> None:
    """Draw KLT motion vectors prev -> cur: survivors of the circular check in green, rejects in red.

    Args:
        prev_left: Previous left image.
        cur_left: Current left image.
        pairs: Previous-frame stereo pairs.
        keep: (N,) circular-check survivors.
        tag: Dataset tag.
        out: Output folder.
    """
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_left, cur_left, pairs.left.reshape(-1, 1, 2), None, winSize=(21, 21), maxLevel=4
    )  # Plain forward KLT (positions of rejects too).
    tracked = tracked.reshape(-1, 2)  # (N, 2).
    ok = status.reshape(-1).astype(bool)  # Forward KLT converged.
    fig, ax = plt.subplots(figsize=(10, 10 * cur_left.shape[0] / cur_left.shape[1] + 0.6))  # Aspect-preserving.
    ax.imshow(cur_left, cmap="gray")  # Current frame.
    for mask, colour, label in ((keep, "lime", "kept (circle closes)"), (ok & ~keep, "red", "rejected by circular L/R check")):  # Two classes.
        ax.plot(
            np.vstack([pairs.left[mask, 0], tracked[mask, 0]]), np.vstack([pairs.left[mask, 1], tracked[mask, 1]]), "-", color=colour, lw=0.8
        )  # Flow vectors.
        ax.scatter(tracked[mask, 0], tracked[mask, 1], s=4, color=colour, label=f"{label}: {int(mask.sum())}")  # End points.
    ax.legend(loc="lower right", fontsize=8, framealpha=0.8)  # Legend.
    ax.set_title(f"{tag}: KLT tracks previous -> current frame")  # Title.
    ax.axis("off")  # No ticks.
    _save(fig, out, f"{tag}_klt_tracks")  # Write.


def main() -> None:
    """Generate all qualitative figures."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # CLI.
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))  # Output folder.
    parser.add_argument("--config", action="append", default=[], help="<config.yaml>:<dataset root>:<frame index>")  # Datasets.
    args = parser.parse_args()  # Parse.
    args.out.mkdir(parents=True, exist_ok=True)  # Ensure the folder exists.
    fig_unc, ax_unc = plt.subplots(figsize=(5.5, 3.8))  # Depth uncertainty figure.
    fig_lm, ax_lm = plt.subplots(figsize=(5.5, 3.8))  # LM convergence figure.
    for spec in args.config:  # One dataset per spec.
        config_path, root, frame_text = spec.rsplit(":", 2)  # Split spec.
        config = ExperimentConfig.from_yaml(config_path).with_overrides(dataset_root=root or None)  # Same settings as the runs.
        dataset = DATASETS.create(config.dataset.name, **config.dataset.params)  # Loader.
        camera, tag, index = dataset.camera, dataset.tag, int(frame_text)  # Shorthands.
        frontend = STEREO_FRONTENDS.create(config.stereo_frontend.name, **config.stereo_frontend.params)  # Front end.
        previous, current = dataset.load_frame(index), dataset.load_frame(index + 1)  # Consecutive frames.

        pairs = frontend.detect(previous.left, previous.right, StereoTracks.empty())  # Stereo pairs in the first frame.
        points, valid = triangulate_rectified(pairs.left, pairs.disparity, camera)  # Depths.
        _stereo_figure(previous.left, previous.right, pairs, points[:, 2], tag, args.out)  # Figure 1.
        cur_pairs, keep = frontend.track((previous.left, previous.right), (current.left, current.right), pairs)  # Circular KLT.
        _klt_figure(previous.left, current.left, pairs, keep, tag, args.out)  # Figure 2.

        cov = triangulation_covariance(pairs.left[valid], pairs.disparity[valid], camera, config.odometry.stereo_sigma_px)  # Point covariances.
        z = np.geomspace(0.3, float(np.nanmax(points[valid, 2])) * 1.2, 200)  # Depth range covering every feature.
        line = ax_unc.plot(
            z,
            np.sqrt(2) * config.odometry.stereo_sigma_px * z**2 / (camera.fx * camera.baseline),
            lw=1.5,
            label=f"{tag}: analytic (B={camera.baseline:.3f} m)",
        )  # sigma_Z curve.
        ax_unc.scatter(
            points[valid, 2], np.sqrt(cov[:, 2, 2]), s=4, alpha=0.5, color=line[0].get_color(), label=f"{tag}: triangulated features"
        )  # Real points.

        observed = cur_pairs.subset(valid[keep])  # Tracks with valid depth.
        pnp = PnPPoseEstimator(PnPConfig(refine="none"))  # RANSAC only.
        estimate = pnp.estimate(points[keep][valid[keep]], observed.left.astype(np.float64), camera.intrinsic_matrix)  # Initial pose.
        cov_kept = triangulation_covariance(
            pairs.left[keep][valid[keep]], pairs.disparity[keep][valid[keep]], camera, config.odometry.stereo_sigma_px
        )  # Their covariances.
        inl = estimate.inlier_mask  # RANSAC inliers.
        rng = np.random.default_rng(0)  # Reproducible perturbations.
        starts = [("RANSAC pose", estimate.camera_from_reference)]  # Start from the RANSAC pose...
        starts += [
            (
                f"perturbed #{k + 1}",
                se3_exp(np.concatenate([rng.normal(scale=0.3, size=3), rng.normal(scale=0.03, size=3)])) @ estimate.camera_from_reference,
            )
            for k in range(3)
        ]  # ...and from perturbed poses.
        for name, start in starts:  # One convergence curve each.
            result = refine_pose_lm(  # Rigid-rig LM.
                start,
                points[keep][valid[keep]][inl],
                observed.left[inl].astype(np.float64),
                camera.intrinsic_matrix,
                observed.right[inl].astype(np.float64),
                camera.baseline,
                cov_kept[inl],
                LMConfig(),
            )
            ax_lm.semilogy(range(len(result.cost_history)), result.cost_history, "o-", ms=3, label=f"{tag}, {name}")  # Cost per accepted step.
        print(f"{tag}: {len(pairs)} stereo pairs, {int(keep.sum())} tracks kept by the circular check, {int(inl.sum())} RANSAC inliers")  # Summary.
    ax_unc.set_xlabel("depth Z [m]")
    ax_unc.set_ylabel(r"$\sigma_Z$ [m]")
    ax_unc.set_yscale("log")
    ax_unc.set_xscale("log")
    ax_unc.set_title(r"Triangulation uncertainty ($\sigma$ = 0.5 px)")
    ax_unc.legend(fontsize=7)
    _save(fig_unc, args.out, "depth_uncertainty")
    ax_lm.set_xlabel("accepted LM step")
    ax_lm.set_ylabel("robust whitened cost")
    ax_lm.set_title("Rigid-rig LM convergence (one frame pair)")
    ax_lm.legend(fontsize=6)
    _save(fig_lm, args.out, "lm_convergence")


if __name__ == "__main__":  # Script entry.
    main()  # Run.
