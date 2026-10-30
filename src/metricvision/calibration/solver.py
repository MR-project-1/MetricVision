"""Stereo calibration (Section 4.1).

Zhang's method (cv2.calibrateCamera) initialises each camera, cv2.stereoCalibrate
initialises T_RL, and our own joint refinement minimises Eq. (2) over both
intrinsics, distortions, the extrinsic and all board poses with a sparse
Levenberg-Marquardt / trust-region solver. The projection model (pinhole +
radial-tangential k1,k2,p1,p2,k3) is implemented here, not taken from OpenCV.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from metricvision.calibration.camera import RawStereoCalib


# ----------------------------------------------------------------------------- model
def project_radtan(X: np.ndarray, intr: np.ndarray, dist: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = intr
    k1, k2, p1, p2, k3 = dist
    x, y = X[:, 0] / X[:, 2], X[:, 1] / X[:, 2]
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([fx * xd + cx, fy * yd + cy], axis=1)


def _rt_apply(rvec, tvec, P):
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64))
    return P @ R.T + tvec


# ----------------------------------------------------------------------------- detection
def board_points(cols: int, rows: int, square: float) -> np.ndarray:
    g = np.zeros((rows * cols, 3), np.float64)
    g[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return g


def find_corners(img: np.ndarray, cols: int, rows: int):
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, c = cv2.findChessboardCornersSB(gray, (cols, rows), flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        ok, c = cv2.findChessboardCorners(gray, (cols, rows), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            return None
        c = cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3))
    return c.reshape(-1, 2).astype(np.float64)


# ----------------------------------------------------------------------------- joint refinement
def _pack(intrL, distL, intrR, distR, rRL, tRL, rvecs, tvecs):
    parts = [intrL, distL, intrR, distR, rRL, tRL]
    for r, t in zip(rvecs, tvecs):
        parts += [r, t]
    return np.concatenate([np.ravel(p) for p in parts])


def _unpack(x, M):
    intrL, distL = x[0:4], x[4:9]
    intrR, distR = x[9:13], x[13:18]
    rRL, tRL = x[18:21], x[21:24]
    poses = x[24:].reshape(M, 6)
    return intrL, distL, intrR, distR, rRL, tRL, poses


def joint_refine(obj: np.ndarray, ptsL, ptsR, init: RawStereoCalib, rvecs, tvecs, verbose: int = 0):
    """Minimise Eq. (2): sum over views/corners of left and right reprojection errors."""
    M, N = len(ptsL), obj.shape[0]
    rRL0, _ = cv2.Rodrigues(init.R)
    x0 = _pack(
        [init.K_L[0, 0], init.K_L[1, 1], init.K_L[0, 2], init.K_L[1, 2]],
        init.D_L,
        [init.K_R[0, 0], init.K_R[1, 1], init.K_R[0, 2], init.K_R[1, 2]],
        init.D_R,
        rRL0.ravel(),
        init.T.ravel(),
        [np.ravel(r) for r in rvecs],
        [np.ravel(t) for t in tvecs],
    )
    uL = np.concatenate(ptsL)
    uR = np.concatenate(ptsR)

    def residuals(x):
        intrL, distL, intrR, distR, rRL, tRL, poses = _unpack(x, M)
        rl, rr = [], []
        for m in range(M):
            XL = _rt_apply(poses[m, :3], poses[m, 3:], obj)
            XR = _rt_apply(rRL, tRL, XL)
            rl.append(project_radtan(XL, intrL, distL))
            rr.append(project_radtan(XR, intrR, distR))
        return np.concatenate([(np.concatenate(rl) - uL).ravel(), (np.concatenate(rr) - uR).ravel()])

    n_res = 4 * M * N
    S = lil_matrix((n_res, x0.size), dtype=np.uint8)
    for m in range(M):
        rows_L = np.arange(2 * m * N, 2 * (m + 1) * N)
        rows_R = rows_L + 2 * M * N
        pose_cols = np.arange(24 + 6 * m, 30 + 6 * m)
        for c in np.concatenate([np.arange(0, 9), pose_cols]):
            S[rows_L, c] = 1
        for c in np.concatenate([np.arange(9, 24), pose_cols]):
            S[rows_R, c] = 1

    sol = least_squares(
        residuals, x0, jac_sparsity=S, method="trf", x_scale="jac", loss="linear", max_nfev=300, ftol=1e-10, xtol=1e-10, verbose=verbose
    )
    intrL, distL, intrR, distR, rRL, tRL, _ = _unpack(sol.x, M)
    K = lambda i: np.array([[i[0], 0, i[2]], [0, i[1], i[3]], [0, 0, 1.0]])
    R, _ = cv2.Rodrigues(rRL)
    res = sol.fun.reshape(-1, 2)
    eL = np.linalg.norm(res[: M * N], axis=1)
    eR = np.linalg.norm(res[M * N :], axis=1)
    out = RawStereoCalib(K(intrL), distL.copy(), K(intrR), distR.copy(), R, tRL.copy(), init.width, init.height)
    report = {
        "rms_left_px": float(np.sqrt(np.mean(eL**2))),
        "rms_right_px": float(np.sqrt(np.mean(eR**2))),
        "rms_joint_px": float(np.sqrt(np.mean(np.concatenate([eL, eR]) ** 2))),
        "n_views": M,
        "n_iterations": int(sol.nfev),
    }
    return out, report


def calibrate_stereo(obj: np.ndarray, ptsL, ptsR, size, refine: bool = True, verbose: int = 0):
    """Full stereo calibration. size = (width, height). Returns (RawStereoCalib, report)."""
    w, h = size
    objs = [obj.astype(np.float32)] * len(ptsL)
    pL = [p.astype(np.float32).reshape(-1, 1, 2) for p in ptsL]
    pR = [p.astype(np.float32).reshape(-1, 1, 2) for p in ptsR]
    flags = 0
    rmsL, KL, DL, rvL, tvL = cv2.calibrateCamera(objs, pL, size, None, None, flags=flags)
    rmsR, KR, DR, _, _ = cv2.calibrateCamera(objs, pR, size, None, None, flags=flags)
    rms_st, KL, DL, KR, DR, R, T, _, _ = cv2.stereoCalibrate(
        objs, pL, pR, KL, DL, KR, DR, size, flags=cv2.CALIB_FIX_INTRINSIC, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    )
    init = RawStereoCalib(KL, DL.ravel()[:5], KR, DR.ravel()[:5], R, T.ravel(), w, h)
    report = {
        "opencv_rms_left_px": float(rmsL),
        "opencv_rms_right_px": float(rmsR),
        "opencv_rms_stereo_px": float(rms_st),
        "opencv_baseline_m": float(np.linalg.norm(T)),
    }
    if not refine:
        return init, report
    calib, rep = joint_refine(obj, ptsL, ptsR, init, rvL, tvL, verbose=verbose)
    report.update(rep)
    report["baseline_m"] = float(np.linalg.norm(calib.T))
    return calib, report


# =============================================================================================
# Partial-view stereo calibration (ChArUco or any target with identified points)
# =============================================================================================
# Each camera may see a different subset of the board in each view, and a view may be seen
# by only one camera. All observations enter one joint problem:
#   params = [fx,fy,cx,cy, k1,k2,p1,p2,k3]_L  [ ... ]_R  [r_RL, t_RL]  [r_v, t_v]_{v=1..M}
#   r = pi_L(T_v X) - u_L          (left observations)
#   r = pi_R(T_RL T_v X) - u_R     (right observations)
# where T_v maps board -> left camera. Unlike cv2.stereoCalibrate this does not need the
# same points in both images, so partially visible boards still constrain T_RL.
from dataclasses import dataclass, field  # noqa: E402

from scipy.spatial.transform import Rotation  # noqa: E402

DIST_MODELS = {  # free entries of (k1, k2, p1, p2, k3)
    "none": [],  # images already undistorted (e.g. RealSense *_rect_raw streams)
    "radial2": [0, 1],
    "radtan4": [0, 1, 2, 3],
    "radtan5": [0, 1, 2, 3, 4],
}
_CV_FLAGS = {
    "none": cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
    "radial2": cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
    "radtan4": cv2.CALIB_FIX_K3,
    "radtan5": 0,
}
PARAM_NAMES = [
    "fx_L",
    "fy_L",
    "cx_L",
    "cy_L",
    "k1_L",
    "k2_L",
    "p1_L",
    "p2_L",
    "k3_L",
    "fx_R",
    "fy_R",
    "cx_R",
    "cy_R",
    "k1_R",
    "k2_R",
    "p1_R",
    "p2_R",
    "k3_R",
    "rx_RL",
    "ry_RL",
    "rz_RL",
    "tx_RL",
    "ty_RL",
    "tz_RL",
]
N_GLOBAL = 24


@dataclass
class StereoView:
    """Observations of one stereo frame. Either side may be empty."""

    name: str
    ids_L: np.ndarray
    uv_L: np.ndarray
    ids_R: np.ndarray
    uv_R: np.ndarray

    @property
    def has_L(self):
        return self.ids_L.size > 0

    @property
    def has_R(self):
        return self.ids_R.size > 0


@dataclass
class CalibResult:
    calib: RawStereoCalib
    report: dict
    views: list
    poses: np.ndarray  # (M,6) board -> left camera, rotvec + t
    obs: dict  # cam, view, ids, uv, residual (px), inlier
    warnings: list = field(default_factory=list)


def _T_from(rvec, tvec):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3))[0]
    T[:3, 3] = np.asarray(tvec, np.float64).reshape(3)
    return T


def _vec_from(T):
    return np.concatenate([cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3]])


def _robust_mean_T(Ts):
    """Chordal rotation mean + median translation, after dropping rotation outliers."""
    Rs = np.stack([T[:3, :3] for T in Ts])
    ts = np.stack([T[:3, 3] for T in Ts])

    def chordal(Rs_):
        U, _, Vt = np.linalg.svd(Rs_.sum(0))
        return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt

    R = chordal(Rs)
    ang = np.array([np.arccos(np.clip((np.trace(R.T @ Ri) - 1) / 2, -1, 1)) for Ri in Rs])
    keep = ang <= 3 * np.median(ang) + np.radians(0.2)
    T = np.eye(4)
    T[:3, :3] = chordal(Rs[keep])
    T[:3, 3] = np.median(ts[keep], axis=0)
    return T, keep


def _stack_obs(views, board_pts, cams=(0, 1)):
    cam, view, ids, uv = [], [], [], []
    for v, sv in enumerate(views):
        for c, i_, u_ in ((0, sv.ids_L, sv.uv_L), (1, sv.ids_R, sv.uv_R)):
            if c in cams and i_.size:
                cam.append(np.full(i_.size, c))
                view.append(np.full(i_.size, v))
                ids.append(i_)
                uv.append(u_)
    cat = (lambda a, dt: np.concatenate(a).astype(dt)) if cam else (lambda a, dt: np.zeros(0, dt))
    ids_ = cat(ids, int)
    return {"cam": cat(cam, int), "view": cat(view, int), "ids": ids_, "uv": np.concatenate(uv) if uv else np.zeros((0, 2)), "obj": board_pts[ids_]}


def _residuals_full(xf, obs):
    intrL, distL, intrR, distR = xf[0:4], xf[4:9], xf[9:13], xf[13:18]
    R_RL = Rotation.from_rotvec(xf[18:21]).as_matrix()
    t_RL = xf[21:24]
    poses = xf[N_GLOBAL:].reshape(-1, 6)
    Rv = Rotation.from_rotvec(poses[:, :3]).as_matrix()
    v = obs["view"]
    X = np.einsum("mij,mj->mi", Rv[v], obs["obj"]) + poses[v, 3:]
    out = np.empty((X.shape[0], 2))
    L = obs["cam"] == 0
    out[L] = project_radtan(X[L], intrL, distL) - obs["uv"][L]
    XR = X[~L] @ R_RL.T + t_RL
    out[~L] = project_radtan(XR, intrR, distR) - obs["uv"][~L]
    return out


def _solve(x_full, free, obs, loss="linear", f_scale=1.0, max_nfev=200):
    """least_squares over the free entries of x_full, with the block-sparse Jacobian."""
    m = obs["cam"].size
    n_full = x_full.size
    S = lil_matrix((2 * m, n_full), dtype=np.uint8)
    rows = np.arange(m)
    for c, cols in ((0, list(range(0, 9))), (1, list(range(9, 24)))):
        r = rows[obs["cam"] == c]
        for col in cols:
            S[2 * r, col] = 1
            S[2 * r + 1, col] = 1
    for k in range(6):
        cols = N_GLOBAL + 6 * obs["view"] + k
        S[2 * rows, cols] = 1
        S[2 * rows + 1, cols] = 1
    S = S.tocsc()[:, free]
    fidx = np.flatnonzero(free)

    def fun(xfree):
        xf = x_full.copy()
        xf[fidx] = xfree
        return _residuals_full(xf, obs).ravel()

    sol = least_squares(
        fun,
        x_full[fidx],
        jac_sparsity=S,
        method="trf",
        x_scale="jac",
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    x = x_full.copy()
    x[fidx] = sol.x
    return x, sol


def _init_camera(views, side, board_pts, size, model):
    sel = [i for i, v in enumerate(views) if (v.has_L if side == 0 else v.has_R)]
    if len(sel) < 3:
        raise ValueError(f"{'left' if side == 0 else 'right'} camera sees the board in only {len(sel)} views")
    objs, imgs = [], []
    for i in sel:
        ids = views[i].ids_L if side == 0 else views[i].ids_R
        uv = views[i].uv_L if side == 0 else views[i].uv_R
        objs.append(board_pts[ids].astype(np.float32))
        imgs.append(uv.astype(np.float32).reshape(-1, 1, 2))
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(objs, imgs, size, None, None, flags=_CV_FLAGS[model])
    D = np.zeros(5) if D is None else np.pad(D.ravel()[:5], (0, max(0, 5 - D.size)))
    poses = {i: _T_from(r, t) for i, r, t in zip(sel, rvecs, tvecs)}
    return K, D, poses, float(rms)


def _pack_global(KL, DL, KR, DR, T_RL):
    g = lambda K: [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
    return np.concatenate([g(KL), DL, g(KR), DR, _vec_from(T_RL)])


def _free_mask(n_views, model, poses_only=False):
    free = np.zeros(N_GLOBAL + 6 * n_views, bool)
    free[N_GLOBAL:] = True
    if not poses_only:
        free[0:4] = free[9:13] = True
        for k in DIST_MODELS[model]:
            free[4 + k] = free[13 + k] = True
        free[18:24] = True
    return free


def _calib_from_x(x, size):
    K = lambda i: np.array([[x[i], 0, x[i + 2]], [0, x[i + 1], x[i + 3]], [0, 0, 1.0]])
    R = Rotation.from_rotvec(x[18:21]).as_matrix()
    return RawStereoCalib(K(0), x[4:9].copy(), K(9), x[13:18].copy(), R, x[21:24].copy(), size[0], size[1])


def _init_poses(views, poses_L, poses_R, T_RL):
    out = np.zeros((len(views), 6))
    for i in range(len(views)):
        if i in poses_L:
            T = poses_L[i]
        else:
            T = np.linalg.inv(T_RL) @ poses_R[i]
        out[i] = _vec_from(T)
    return out


def _rms(e):
    return float(np.sqrt(np.mean(e**2))) if e.size else float("nan")


def rectification_error(calib: RawStereoCalib, views, size):
    """|v_L - v_R| of the same corner after undistortion + rectification (px)."""
    R1, R2, P1, P2, *_ = cv2.stereoRectify(
        calib.K_L, calib.D_L, calib.K_R, calib.D_R, size, calib.R, calib.T.reshape(3, 1), flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    dv = []
    for v in views:
        common, iL, iR = np.intersect1d(v.ids_L, v.ids_R, return_indices=True)
        if common.size == 0:
            continue
        pL = cv2.undistortPoints(v.uv_L[iL].reshape(-1, 1, 2), calib.K_L, calib.D_L, R=R1, P=P1).reshape(-1, 2)
        pR = cv2.undistortPoints(v.uv_R[iR].reshape(-1, 1, 2), calib.K_R, calib.D_R, R=R2, P=P2).reshape(-1, 2)
        dv.append(pL[:, 1] - pR[:, 1])
    dv = np.concatenate(dv) if dv else np.zeros(0)
    if dv.size == 0:
        return {"n": 0}
    a = np.abs(dv)
    return {
        "n": int(dv.size),
        "mean_abs_px": float(a.mean()),
        "median_abs_px": float(np.median(a)),
        "p95_abs_px": float(np.percentile(a, 95)),
        "max_abs_px": float(a.max()),
    }


def _coverage(uv, size, grid=(8, 6)):
    w, h = size
    if uv.size == 0:
        return 0.0
    gx = np.clip((uv[:, 0] * grid[0] / w).astype(int), 0, grid[0] - 1)
    gy = np.clip((uv[:, 1] * grid[1] / h).astype(int), 0, grid[1] - 1)
    return float(np.unique(gy * grid[0] + gx).size / (grid[0] * grid[1]))


def calibrate_stereo_views(
    views: list, board_pts: np.ndarray, size, model: str = "radtan4", holdout: float = 0.0, drop_bad_views: bool = True, verbose: bool = False
) -> CalibResult:
    """Joint stereo calibration from partially visible identified targets (e.g. ChArUco)."""
    if model not in DIST_MODELS:
        raise ValueError(f"model must be one of {list(DIST_MODELS)}")
    all_views = list(views)
    test_views = []
    if holdout > 0:
        step = max(2, int(round(1 / holdout)))
        test_views = [v for i, v in enumerate(all_views) if i % step == step - 1]
        views = [v for i, v in enumerate(all_views) if i % step != step - 1]
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    # ---- 1. per-camera Zhang initialisation (OpenCV), varying point subsets per view
    KL, DL, PL, rmsL0 = _init_camera(views, 0, board_pts, size, model)
    KR, DR, PR, rmsR0 = _init_camera(views, 1, board_pts, size, model)
    both = [i for i in PL if i in PR]
    if len(both) < 3:
        raise ValueError(f"only {len(both)} views where both cameras see the board; need >= 3 (aim for 20+)")
    T_RL0, keep = _robust_mean_T([PR[i] @ np.linalg.inv(PL[i]) for i in both])
    log(f"init: rms L {rmsL0:.3f}  R {rmsR0:.3f} px; baseline {1000 * np.linalg.norm(T_RL0[:3, 3]):.2f} mm from {keep.sum()}/{len(both)} views")

    # ---- 2. joint robust refinement, outlier and bad-view rejection, final linear refinement
    x = np.concatenate([_pack_global(KL, DL, KR, DR, T_RL0), _init_poses(views, PL, PR, T_RL0).ravel()])
    obs = _stack_obs(views, board_pts)
    free = _free_mask(len(views), model)
    x, _ = _solve(x, free, obs, loss="huber", f_scale=1.0)
    e = np.linalg.norm(_residuals_full(x, obs), axis=1)
    thr = max(1.0, 5.0 * float(np.median(e)))
    inl = e < thr
    dropped = []
    if drop_bad_views:
        vr = np.array([_rms(e[(obs["view"] == v) & inl]) for v in range(len(views))])
        bad = vr > max(1.0, 3.0 * np.nanmedian(vr))
        dropped = [views[v].name for v in np.flatnonzero(bad)]
        inl &= ~bad[obs["view"]]
    sub = {k: val[inl] for k, val in obs.items()}
    x, sol = _solve(x, free, sub, loss="linear")
    r = _residuals_full(x, sub)
    e_in = np.linalg.norm(r, axis=1)
    calib = _calib_from_x(x, size)

    # ---- 3. formal uncertainty: Sigma = s^2 (J^T J)^-1 on the free parameters
    J = sol.jac
    JTJ = (J.T @ J).toarray() if hasattr(J, "toarray") else J.T @ J
    dof = max(r.size - int(free.sum()), 1)
    s2 = float(np.sum(r**2) / dof)
    try:
        cov = s2 * np.linalg.inv(JTJ)
    except np.linalg.LinAlgError:
        cov = s2 * np.linalg.pinv(JTJ)
    fidx = np.flatnonzero(free)
    std = np.full(x.size, np.nan)
    std[fidx] = np.sqrt(np.maximum(np.diag(cov), 0))
    pos = {p: k for k, p in enumerate(fidx)}
    t = x[21:24]
    g = t / np.linalg.norm(t)
    ct = cov[np.ix_([pos[i] for i in range(21, 24)], [pos[i] for i in range(21, 24)])]
    cr = cov[np.ix_([pos[i] for i in range(18, 21)], [pos[i] for i in range(18, 21)])]
    unc = {name: float(std[i]) for i, name in enumerate(PARAM_NAMES) if np.isfinite(std[i])}
    unc["baseline_mm"] = float(1000 * np.sqrt(g @ ct @ g))
    unc["rotation_deg"] = float(np.degrees(np.sqrt(np.trace(cr))))

    # ---- 4. diagnostics
    cam = sub["cam"]
    R_RL = calib.R
    tilt_L, tilt_R, dist_v = [], [], []
    for v in range(len(views)):
        Rv = Rotation.from_rotvec(x[N_GLOBAL + 6 * v : N_GLOBAL + 6 * v + 3]).as_matrix()
        tilt_L.append(np.degrees(np.arccos(abs(Rv[2, 2]))))
        tilt_R.append(np.degrees(np.arccos(abs((R_RL @ Rv)[2, 2]))))
        dist_v.append(float(np.linalg.norm(x[N_GLOBAL + 6 * v + 3 : N_GLOBAL + 6 * v + 6])))
    per_view = []
    for v, sv in enumerate(views):
        m = sub["view"] == v
        per_view.append(
            {
                "name": sv.name,
                "n_L": int((m & (cam == 0)).sum()),
                "n_R": int((m & (cam == 1)).sum()),
                "rms_L": _rms(e_in[m & (cam == 0)]),
                "rms_R": _rms(e_in[m & (cam == 1)]),
                "tilt_deg": float(tilt_L[v]),
                "distance_m": dist_v[v],
            }
        )
    used = [sv for sv in views if sv.name not in dropped]
    report = {
        "distortion_model": model,
        "n_views": len(views),
        "n_views_both": int(sum(v.has_L and v.has_R for v in views)),
        "n_views_left_only": int(sum(v.has_L and not v.has_R for v in views)),
        "n_views_right_only": int(sum(v.has_R and not v.has_L for v in views)),
        "dropped_views": dropped,
        "n_observations": int(obs["cam"].size),
        "n_outlier_observations": int((~inl).sum()),
        "outlier_threshold_px": float(thr),
        "opencv_init_rms_left_px": rmsL0,
        "opencv_init_rms_right_px": rmsR0,
        "rms_left_px": _rms(e_in[cam == 0]),
        "rms_right_px": _rms(e_in[cam == 1]),
        "rms_joint_px": _rms(e_in),
        "baseline_m": float(np.linalg.norm(calib.T)),
        "uncertainty_1sigma": unc,
        "rectification_check": rectification_error(calib, used, size),
        "coverage_left": _coverage(sub["uv"][cam == 0], size),
        "coverage_right": _coverage(sub["uv"][cam == 1], size),
        "tilt_deg": {"min": float(np.min(tilt_L)), "median": float(np.median(tilt_L)), "max": float(np.max(tilt_L))},
        "distance_m": {"min": float(np.min(dist_v)), "median": float(np.median(dist_v)), "max": float(np.max(dist_v))},
        "per_view": per_view,
    }

    # ---- 5. hold-out: board poses re-estimated with the calibration frozen
    if test_views:
        tobs = _stack_obs(test_views, board_pts)
        tp = np.zeros((len(test_views), 6))
        for k, tv in enumerate(test_views):
            if tv.has_L:
                ok, rv, tt = cv2.solvePnP(board_pts[tv.ids_L], tv.uv_L, calib.K_L, calib.D_L)
                tp[k] = _vec_from(_T_from(rv, tt))
            else:
                ok, rv, tt = cv2.solvePnP(board_pts[tv.ids_R], tv.uv_R, calib.K_R, calib.D_R)
                T_RL = np.eye(4)
                T_RL[:3, :3], T_RL[:3, 3] = calib.R, calib.T
                tp[k] = _vec_from(np.linalg.inv(T_RL) @ _T_from(rv, tt))
        xt = np.concatenate([x[:N_GLOBAL], tp.ravel()])
        xt, _ = _solve(xt, _free_mask(len(test_views), model, poses_only=True), tobs, loss="huber", f_scale=1.0)
        et = np.linalg.norm(_residuals_full(xt, tobs), axis=1)
        et = et[et < thr]
        report["holdout"] = {"n_views": len(test_views), "rms_px": _rms(et), "ratio_to_train": _rms(et) / max(report["rms_joint_px"], 1e-9)}

    warnings = _calibration_warnings(report)
    report["warnings"] = warnings
    full_e = np.linalg.norm(_residuals_full(x, obs), axis=1)
    obs_out = dict(obs)
    obs_out["residual_px"] = full_e
    obs_out["inlier"] = inl
    return CalibResult(calib, report, views, x[N_GLOBAL:].reshape(-1, 6), obs_out, warnings)


def _calibration_warnings(rep: dict) -> list:
    w = []
    if rep["rms_joint_px"] > 0.5:
        w.append(f"joint RMS {rep['rms_joint_px']:.2f} px > 0.5 px: check board flatness, focus, motion blur")
    if min(rep["coverage_left"], rep["coverage_right"]) < 0.6:
        w.append("board corners cover < 60 % of an image: move the board into the corners and edges (distortion is poorly constrained there)")
    if rep["tilt_deg"]["median"] < 15 or rep["tilt_deg"]["max"] < 30:
        w.append("mostly fronto-parallel views: focal length and principal point are poorly constrained (tilt the board 30-45 deg about both axes)")
    if rep["n_views_both"] < 15:
        w.append(f"only {rep['n_views_both']} views seen by both cameras; aim for 20-40")
    rc = rep["rectification_check"]
    if rc.get("n", 0) and rc["mean_abs_px"] > 0.3:
        w.append(f"rectified corners differ by {rc['mean_abs_px']:.2f} px vertically (target < 0.3)")
    if "holdout" in rep and rep["holdout"]["ratio_to_train"] > 1.5:
        w.append("hold-out RMS is much larger than training RMS: overfitting - use fewer distortion terms or more varied views")
    u = rep["uncertainty_1sigma"]
    if u["rotation_deg"] > 0.05 or u["baseline_mm"] > 0.5:
        w.append(
            f"extrinsic uncertainty {u['rotation_deg']:.3f} deg / {u['baseline_mm']:.2f} mm (1 sigma) is "
            "large compared with the errors that already hurt rigid VO (0.1 deg / 1 mm)"
        )
    return w
