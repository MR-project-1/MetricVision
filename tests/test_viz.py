"""Tests for the VO debug view and the per-frame debug records of both pipelines."""

from __future__ import annotations

import numpy as np

from metricvision.backend.odometry import TrackingResult
from metricvision.backend.stats import FrameStats
from metricvision.core.geometry import make_pose
from metricvision.core.types import FrameDebug
from metricvision.viz import VOVisualizer
from metricvision.viz.frame_view import depth_colours


def _result(index: int, x: float, debug: FrameDebug | None, ok: bool = True) -> TrackingResult:
    stats = FrameStats(index, 0.1 * index, 10, 8, 6 if ok else 0, float("nan"), ok, False, 1.0, 1.0, 0.0, 2.0)
    return TrackingResult(make_pose(np.eye(3), [0.0, 0.0, x]), stats, debug)  # Camera z = forward.


def _debug(reference_index: int, stereo: bool) -> FrameDebug:
    rng = np.random.default_rng(0)
    cur = rng.uniform([20, 20], [300, 220], (10, 2)).astype(np.float32)
    pts = np.column_stack([np.zeros(10), np.zeros(10), np.linspace(1, 10, 10)])
    pts[0] = np.nan  # One track without depth.
    inliers = np.ones(10, dtype=bool)
    inliers[[0, 1]] = False  # Plus one outlier.
    right = (cur - [15, 0]).astype(np.float32) if stereo else None
    return FrameDebug(reference_index, cur + 1, cur, right, pts, inliers)


def test_depth_colours_are_ordered_and_handle_nan() -> None:
    c = depth_colours(np.array([0.5, 15.0, np.nan]))
    assert c.shape == (3, 3)
    assert c[0, 2] > c[0, 0] and c[1, 0] > c[1, 2]  # Near is red (BGR: R last), far is blue.
    assert tuple(c[2]) == (150, 150, 150)  # Grey for no depth.


def test_visualizer_renders_sparse_and_dense_frames() -> None:
    body_from_camera = make_pose(np.array([[0.0, 0, 1], [-1, 0, 0], [0, -1, 0]]), [0.3, 0, 0.5])  # Optical -> base axes.
    left = np.full((240, 320), 100, np.uint8)
    right = np.full((240, 320), 90, np.uint8)
    for stereo in (True, False):
        viz = VOVisualizer(body_from_camera, "test")
        first = viz.render(0, left, right, _result(0, 0.0, None), np.eye(4))
        canvas = viz.render(1, left, right, _result(1, 0.5, _debug(0, stereo)), make_pose(np.eye(3), [0.5, 0, 0]))
        assert first.shape == canvas.shape == (240 + 360, 640, 3)
        assert canvas.dtype == np.uint8
        lost = viz.render(2, left, right, _result(2, 0.5, _debug(1, stereo), ok=False))
        assert lost.shape == canvas.shape
        np.testing.assert_allclose(viz._vo[1][:3, 3], [0.5, 0.0, 0.0], atol=1e-12)  # Camera z step -> base x step.
        assert viz._failed == [2]
