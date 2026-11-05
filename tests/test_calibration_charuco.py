"""ChArUco stereo calibration: detection, partial / one-camera views, recovery of a known rig."""

import numpy as np
import pytest

from metricvision.calibration._geometry import make_T, project, rot_angle, so3_exp
from metricvision.calibration.charuco import EMPTY, BoardSpec, CharucoDetector, Detection, select_views
from metricvision.calibration.charuco_synth import CharucoRenderer, default_rig, random_board_pose
from metricvision.calibration.solver import StereoView, calibrate_stereo_views

SPEC = BoardSpec()


def test_spec_validation():
    with pytest.raises(ValueError):
        BoardSpec(squares_x=12, squares_y=12, dictionary="DICT_4X4_50").validate()  # 72 markers > 50
    with pytest.raises(ValueError):
        BoardSpec(marker_length=0.05).validate()
    pts = SPEC.corner_points()
    assert pts.shape == (40, 3) and np.allclose(pts[0], [0.04, 0.04, 0]) and np.allclose(pts[8], [0.04, 0.08, 0])


@pytest.mark.parametrize("legacy", [False, True])
def test_detection_accuracy_and_legacy_autodetect(legacy):
    K = np.array([[450.0, 0, 320], [0, 450, 240], [0, 0, 1]])
    D = np.zeros(5)
    rend = CharucoRenderer(SPEC, legacy=legacy)
    det = CharucoDetector(SPEC)  # legacy: auto
    T = make_T(so3_exp([0.2, -0.15, 0.05]), [-0.17, -0.11, 0.6])
    img = rend.render(K, D, T, (640, 480), rng=np.random.default_rng(0))
    d = det.detect(img)
    assert d.n == SPEC.n_corners
    assert det.legacy == legacy
    gt = project(K, SPEC.corner_points()[d.ids] @ T[:3, :3].T + T[:3, 3])
    assert np.median(np.linalg.norm(d.uv - gt, axis=1)) < 0.15


def test_collinear_views_rejected():
    det = CharucoDetector(SPEC, min_corners=6)
    row = np.arange(8)  # one row of corners -> homography degenerate
    assert not det.usable(Detection(row, np.random.rand(8, 2)))
    two_rows = np.concatenate([np.arange(8), np.arange(8, 16)])
    assert det.usable(Detection(two_rows, np.random.rand(16, 2)))


def test_view_selection_spreads_out():
    feats = [np.array([0.0, 0, 0, 0, 0])] * 10 + [np.array([1.0, 1, 1, 1, 1]), np.array([-1.0, 0, 0, 0, 0])]
    sel = select_views(feats, [1] * 12, 3)
    assert 10 in sel and 11 in sel


@pytest.fixture(scope="module")
def stereo_views():
    KL, DL, KR, DR, T_RL, size = default_rig()
    rend = CharucoRenderer(SPEC)
    det = CharucoDetector(SPEC, min_corners=10)
    rng = np.random.default_rng(11)
    views = []
    for i in range(28):
        T = random_board_pose(rng, SPEC)
        dl = det.detect(rend.render(KL, DL, T, size, rng=rng))
        dr = det.detect(rend.render(KR, DR, T_RL @ T, size, rng=rng))
        if i in (3, 9):  # board hidden from the right camera
            dr = EMPTY
        if i == 14:  # ... and from the left camera
            dl = EMPTY
        if dl.n or dr.n:
            views.append(StereoView(f"v{i}", dl.ids, dl.uv, dr.ids, dr.uv))
    return views, (KL, DL, KR, DR, T_RL, size)


def test_partial_view_stereo_calibration(stereo_views):
    views, (KL, DL, KR, DR, T_RL, size) = stereo_views
    assert any(0 < v.ids_L.size < SPEC.n_corners for v in views)  # partial boards exercised
    res = calibrate_stereo_views(views, SPEC.corner_points(), size, "radtan4", holdout=0.2)
    c, r = res.calib, res.report
    assert r["n_views_left_only"] + r["n_views_right_only"] >= 1
    assert r["rms_joint_px"] < 0.4
    e_b = 1000 * abs(np.linalg.norm(c.T) - np.linalg.norm(T_RL[:3, 3]))
    e_r = np.degrees(rot_angle(c.R @ T_RL[:3, :3].T))
    u = r["uncertainty_1sigma"]
    assert e_b < 0.3 and e_b < 4 * u["baseline_mm"] + 0.05, (e_b, u["baseline_mm"])
    assert e_r < 0.12 and e_r < 4 * u["rotation_deg"] + 0.01, (e_r, u["rotation_deg"])
    assert abs(c.K_L[0, 0] - KL[0, 0]) < 2.0 and abs(c.K_R[0, 2] - KR[0, 2]) < 3.0
    assert np.allclose(c.D_L[:2], DL[:2], atol=0.02)
    assert r["rectification_check"]["mean_abs_px"] < 0.3
    assert "holdout" in r and r["holdout"]["ratio_to_train"] < 1.5


def test_no_distortion_model_keeps_zero(stereo_views):
    views, (_, _, _, _, _, size) = stereo_views
    res = calibrate_stereo_views(views[:12], SPEC.corner_points(), size, "none", drop_bad_views=False)
    assert np.all(res.calib.D_L == 0) and np.all(res.calib.D_R == 0)
