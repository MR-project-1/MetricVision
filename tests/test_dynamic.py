"""Tests for dynamic-object handling: mask lookups and the geometric dynamic/occlusion gate."""

from __future__ import annotations  # Postponed evaluation of type hints.

from dataclasses import replace  # Config variants.
from pathlib import Path  # Temporary output folder.

import numpy as np  # Synthetic data.

from metricvision.backend.odometry import OdometryConfig  # Back-end settings.
from metricvision.backend.pose import PnPConfig  # Refinement selection.
from metricvision.backend.stereo_odometry import SparseStereoOdometry  # Unit under test.
from metricvision.config import ComponentConfig, ExperimentConfig  # Programmatic configs.
from metricvision.core.types import StereoCamera  # Camera model.
from metricvision.dynamic import pixels_in_mask  # Unit under test.
from metricvision.frontend.stereo_klt import FastOrbKltFrontend, StereoTracks  # Front end and tracks.
from metricvision.pipeline import ExperimentRunner  # End-to-end runner.

CAMERA = StereoCamera(fx=718.856, fy=718.856, cx=607.19, cy=185.22, baseline=0.537, width=1241, height=376)  # KITTI-like rig.


def test_pixels_in_mask_handles_subpixel_and_out_of_image_points() -> None:
    """Points are looked up at their nearest pixel; points outside the image count as unmasked."""
    mask = np.zeros((10, 20), dtype=bool)  # Small mask.
    mask[2:5, 3:8] = True  # A masked box.
    points = np.array([[4.4, 3.2], [8.6, 3.0], [-1.0, 3.0], [5.0, 50.0]])  # Inside, just outside, left of image, below image.
    np.testing.assert_array_equal(pixels_in_mask(mask, points), [True, False, False, False])  # Lookups.
    assert pixels_in_mask(mask, np.zeros((0, 2))).shape == (0,)  # Empty input.


def _tracks_and_anchors(n: int = 50) -> tuple[StereoTracks, np.ndarray, np.ndarray, SparseStereoOdometry]:
    """Stereo pairs of static points seen from the origin, with their exact world anchors.

    Args:
        n: Number of points.

    Returns:
        Tracks, anchor points, anchor covariances and an odometry object (identity pose).
    """
    rng = np.random.default_rng(0)  # Seeded generator.
    u, v, z = rng.uniform(100, 1100, n), rng.uniform(50, 320, n), rng.uniform(5, 30, n)  # Pixels and depths.
    d = CAMERA.fx * CAMERA.baseline / z  # Disparities.
    tracks = StereoTracks(np.column_stack([u, v]).astype(np.float32), np.column_stack([u - d, v]).astype(np.float32))  # Pairs.
    odometry = SparseStereoOdometry(CAMERA, FastOrbKltFrontend(), OdometryConfig(sparse_point_anchor="fused", dynamic_gate=True))  # Gate on.
    points, cov = odometry._to_world(tracks, np.eye(4))  # Exact anchors (world = camera).
    return tracks, points, cov, odometry


def test_gate_rejects_moved_points_and_keeps_static_ones() -> None:
    """Landmarks displaced by 1 m (a moving car) fail the Mahalanobis test; static ones pass."""
    tracks, points, cov, odometry = _tracks_and_anchors()  # Static scene.
    moved = np.arange(10)  # The first ten points belong to a moving object...
    displaced = points.copy()  # ...whose stored landmarks are now 1 m away from where stereo sees them.
    displaced[moved, 0] += 1.0  # Lateral motion.
    _, _, consistent = odometry._fuse((displaced, cov), tracks, np.eye(4))  # Gate.
    assert not consistent[moved].any()  # Moving points rejected.
    assert consistent[10:].all()  # Static points kept.


def test_sparse_pipeline_with_dynamic_gate_tracks_static_scene(tmp_path: Path) -> None:
    """On a static scene the gate must not hurt tracking.

    Args:
        tmp_path: Pytest-provided temporary folder.
    """
    config = ExperimentConfig(  # Synthetic sequence, sparse pipeline, gate on.
        dataset=ComponentConfig("synthetic", {"num_frames": 30}),
        matcher=ComponentConfig("orb"),
        depth=ComponentConfig("sgbm", {"num_disparities": 32, "min_depth": 0.5, "max_depth": 20.0}),
        output_dir=tmp_path,
        pipeline="sparse",
        run_name="gate",
    )
    config = replace(config, odometry=OdometryConfig(pnp=PnPConfig(refine="lm"), sparse_point_anchor="fused", dynamic_gate=True))  # Gate on.
    summary = ExperimentRunner(config).run(show_progress=False)  # Run.
    assert summary.num_tracking_failures == 0  # Never lose track.
    assert summary.errors is not None and summary.errors.ate_rmse_m < 0.02  # Accurate.
