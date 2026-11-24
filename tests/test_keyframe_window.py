"""Tests for the keyframe window's monocular depth-prior gating."""

from __future__ import annotations  # Postponed evaluation of type hints.

import numpy as np  # Synthetic data.

from metricvision.backend.keyframe_window import DepthPriorConfig, KeyframeWindow  # Unit under test.
from metricvision.core.stereo import triangulation_covariance  # Realistic stereo uncertainty.
from metricvision.core.types import StereoCamera  # Camera model.

CAMERA = StereoCamera(fx=718.856, fy=718.856, cx=607.19, cy=185.22, baseline=0.537, width=1241, height=376)  # KITTI-like rig.


def _scene(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Landmarks in front of an identity-pose camera with stereo observations and covariances.

    Args:
        seed: Random seed.

    Returns:
        ids, pixels (u_l, v_l, u_r), world points, covariances.
    """
    rng = np.random.default_rng(seed)  # Seeded generator.
    n = 400  # Landmarks.
    u = rng.uniform(50, CAMERA.width - 50, n)  # Columns.
    v = rng.uniform(50, CAMERA.height - 50, n)  # Rows.
    z = rng.uniform(4, 60, n)  # Depths (near ones give trusted stereo for the scale check).
    d = CAMERA.fx * CAMERA.baseline / z  # Disparities.
    points = np.column_stack([(u - CAMERA.cx) * z / CAMERA.fx, (v - CAMERA.cy) * z / CAMERA.fy, z])  # World = camera.
    cov = triangulation_covariance(np.column_stack([u, v]), d, CAMERA, 0.5)  # Stereo covariance.
    return np.arange(n), np.column_stack([u, v, u - d]), points, cov  # Keyframe inputs.


def _depth_map(points: np.ndarray, pixels: np.ndarray, scale: float) -> np.ndarray:
    """A network depth map that equals ``scale`` x true depth at every observed pixel.

    Args:
        points: World points (camera frame here).
        pixels: Observations.
        scale: Network scale error.

    Returns:
        (H, W) float32 depth map.
    """
    depth = np.full((CAMERA.height, CAMERA.width), np.nan, dtype=np.float32)  # Empty map.
    depth[np.rint(pixels[:, 1]).astype(int), np.rint(pixels[:, 0]).astype(int)] = scale * points[:, 2]  # Values at the observations.
    return depth  # Map.


def test_scale_alignment_recovers_mis_scaled_network() -> None:
    """A network that is 30% off in scale is aligned to stereo and its priors are accepted."""
    ids, pixels, points, cov = _scene(0)  # Scene.
    window = KeyframeWindow(5, prior_config=DepthPriorConfig(scale_align=True, min_stereo_rel_sigma=0.0))  # Aligning window (no weak-stereo gate).
    stats = window.add_keyframe(0, np.eye(4), ids, pixels, points, cov, _depth_map(points, pixels, 1.3))  # Insert.
    assert abs(stats.prior_scale - 1 / 1.3) < 0.01  # Measured scale ratio.
    assert stats.priors > 350  # Nearly all priors survive after alignment.


def test_inconsistent_priors_are_rejected() -> None:
    """Priors that disagree strongly with precise stereo depth fail the consistency gate."""
    ids, pixels, points, cov = _scene(1)  # Scene.
    depth = _depth_map(points, pixels, 1.0)  # Correct network.
    near = points[:, 2] < 15  # Precise stereo region.
    corrupt = np.flatnonzero(near)[:30]  # Corrupt 30 near observations.
    depth[np.rint(pixels[corrupt, 1]).astype(int), np.rint(pixels[corrupt, 0]).astype(int)] *= 2.0  # Wrong by 2x.
    window = KeyframeWindow(5, prior_config=DepthPriorConfig(scale_align=True, min_stereo_rel_sigma=0.0))  # Window (no weak-stereo gate).
    window.add_keyframe(0, np.eye(4), ids, pixels, points, cov, depth)  # Insert.
    kept = np.isfinite(window._keyframes[0].prior_depth)  # Accepted priors.
    assert not kept[corrupt].any()  # Corrupted priors rejected.
    assert kept.sum() > 300  # The rest survive.


def test_unaligned_mis_scaled_keyframe_is_dropped() -> None:
    """Without alignment, a keyframe whose network scale is off by more than the gate gets no priors."""
    ids, pixels, points, cov = _scene(2)  # Scene.
    window = KeyframeWindow(5, prior_config=DepthPriorConfig(scale_align=False, scale_gate=0.2))  # Gating only.
    stats = window.add_keyframe(0, np.eye(4), ids, pixels, points, cov, _depth_map(points, pixels, 1.5))  # 50% off.
    assert stats.priors == 0  # Whole keyframe rejected.
