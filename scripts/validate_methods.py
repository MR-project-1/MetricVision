"""Controlled validation of the method upgrades (no dataset download needed).

Writes figures (PDF) and a JSON of numbers used in the revised mid-report:
  1. LM consistency (NEES) with a shared vs a mismatched noise model;
  2. stereo vs OpenCV RANSAC with left- and right-only outliers;
  3. XYZ vs inverse-depth landmark models at KITTI ranges (bias, consistency);
  4. anchor fusion with unmodelled pose error, with and without pose inflation;
  5. chained-KLT drift vs birth-template re-alignment.

Usage: uv run python scripts/validate_methods.py --out report/validation
"""

from __future__ import annotations  # Postponed evaluation of type hints.

import argparse  # CLI.
import json  # Results.
import time  # Timing.
from pathlib import Path  # Output folder.

import cv2  # KLT, images.
import matplotlib  # Plots.

matplotlib.use("Agg")  # Headless backend.
import matplotlib.pyplot as plt  # noqa: E402  # Plotting.
import numpy as np  # noqa: E402  # Maths.
from scipy import stats  # noqa: E402  # chi^2 quantiles.

from metricvision.backend.anchors import fuse_xyz, pose_uncertainty_covariance  # noqa: E402  # Fusion models.
from metricvision.backend.lm import LMConfig, refine_pose_lm  # noqa: E402  # Rig LM.
from metricvision.backend.pose import PnPConfig, PnPPoseEstimator  # noqa: E402  # RANSAC variants.
from metricvision.core.geometry import invert_pose, make_pose, se3_exp  # noqa: E402  # Poses.
from metricvision.core.stereo import triangulation_covariance  # noqa: E402  # Covariance.
from metricvision.core.types import StereoCamera  # noqa: E402  # Camera.
from metricvision.frontend.alignment import align_warped, predict_affine  # noqa: E402  # Re-alignment.

CAM = StereoCamera(fx=718.856, fy=718.856, cx=607.19, cy=185.22, baseline=0.537, width=1241, height=376)  # KITTI 00 rectified.
K = CAM.intrinsic_matrix  # Intrinsics.
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7, "figure.dpi": 150})  # Small figures.


def scene(rng: np.random.Generator, n: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """KITTI-like scene: points 5-80 m ahead, car-like motion (0.9 m forward, small yaw).

    Args:
        rng: Generator.
        n: Points.

    Returns:
        Points (reference frame), true T_cur_ref, exact left and right pixels.
    """
    while True:  # Rejection-sample points visible in the current image.
        z = rng.uniform(5, 80, 4 * n)
        pts = np.column_stack([rng.uniform(-0.8, 0.8, 4 * n) * z, rng.uniform(-0.2, 0.25, 4 * n) * z, z])
        pose = se3_exp(np.array([0.02, 0.01, -0.9, 0.001, 0.004, 0.0005]))  # T_cur_ref.
        p = pts @ pose[:3, :3].T + pose[:3, 3]
        u = CAM.fx * p[:, 0] / p[:, 2] + CAM.cx
        v = CAM.fy * p[:, 1] / p[:, 2] + CAM.cy
        ok = (p[:, 2] > 1) & (u > 5) & (u < CAM.width - 5) & (v > 5) & (v < CAM.height - 5) & (u - CAM.fx * CAM.baseline / p[:, 2] > 5)
        if ok.sum() >= n:
            idx = np.flatnonzero(ok)[:n]
            left = np.column_stack([u[idx], v[idx]])
            right = left - np.column_stack([CAM.fx * CAM.baseline / p[idx, 2], np.zeros(n)])
            return pts[idx], pose, left, right


def pose_error_vec(est: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Left-perturbation error (rho, phi) to first order.

    Args:
        est: Estimate.
        truth: Truth.

    Returns:
        (6,) error.
    """
    delta = est @ invert_pose(truth)
    rvec, _ = cv2.Rodrigues(delta[:3, :3])
    return np.concatenate([delta[:3, 3], rvec.ravel()])


def exp_nees(rng: np.random.Generator, trials: int = 300) -> dict:
    """NEES of the rig LM: true noise 0.5 px, LM assuming 0.5 px (shared) or 1.0 px (mid-report).

    Args:
        rng: Generator.
        trials: Monte Carlo trials.

    Returns:
        Results.
    """
    sigma = 0.5  # True pixel noise.
    out = {}
    for name, lm_sigma in (("shared_0.5", 0.5), ("mismatched_1.0", 1.0)):
        nees, chi2_in = [], []
        for _ in range(trials):
            pts, truth, left0, right0 = scene(rng, 300)
            left = left0 + rng.normal(0, sigma, left0.shape)
            right = right0.copy()
            right[:, 0] += rng.normal(0, sigma, len(right))
            res = refine_pose_lm(truth, pts, left, K, right, CAM.baseline, None, LMConfig(pixel_sigma=lm_sigma, huber_delta=np.inf))
            e = pose_error_vec(res.camera_from_reference, truth)
            nees.append(float(e @ np.linalg.solve(res.covariance, e)))
            chi2_in.append(float(np.mean(res.inlier_mask)))
        out[name] = {"nees": nees, "mean_nees": float(np.mean(nees)), "inlier_fraction": float(np.mean(chi2_in))}
    return out


def exp_ransac(rng: np.random.Generator, trials: int = 100) -> dict:
    """OpenCV vs stereo RANSAC (both + rig LM) with 20 % left outliers and 10 % right-only outliers.

    Args:
        rng: Generator.
        trials: Trials.

    Returns:
        Results.
    """
    out = {}
    data = []
    for _ in range(trials):  # Same data for both estimators.
        pts, truth, left, right = scene(rng, 400)
        left = left + rng.normal(0, 0.5, left.shape)
        right = right + rng.normal(0, 0.5, right.shape)
        n_l, n_r = 80, 40
        left[:n_l] += rng.uniform(10, 40, (n_l, 2)) * rng.choice([-1, 1], (n_l, 2))
        right[n_l : n_l + n_r, 0] += rng.uniform(5, 20, n_r) * rng.choice([-1, 1], n_r)  # Wrong stereo match only.
        prior = se3_exp(np.array([0.05, -0.03, 0.08, 0.002, -0.003, 0.001])) @ truth  # Motion-model prior with error.
        data.append((pts, truth, left, right, prior))
    for name in ("opencv", "stereo"):
        est = PnPPoseEstimator(PnPConfig(ransac=name, refine="none", reprojection_error_px=2.0, iterations=200))
        est_lm = PnPPoseEstimator(PnPConfig(ransac=name, refine="lm", reprojection_error_px=2.0, iterations=200))
        right_pass, t_err, r_err, ms = [], [], [], []
        for pts, truth, left, right, prior in data:
            start = time.perf_counter()
            raw = est.estimate(pts, left, K, prior, points_2d_right=right, baseline=CAM.baseline)
            ms.append((time.perf_counter() - start) * 1000)
            right_pass.append(float(raw.inlier_mask[80:120].mean()))
            ref = est_lm.estimate(pts, left, K, prior, points_2d_right=right, baseline=CAM.baseline, pixel_sigma=0.5)
            e = pose_error_vec(ref.camera_from_reference, truth)
            t_err.append(float(np.linalg.norm(e[:3]) * 100))
            r_err.append(float(np.degrees(np.linalg.norm(e[3:]))))
        out[name] = {
            "right_only_outliers_accepted_pct": 100 * float(np.mean(right_pass)),
            "t_err_cm_median": float(np.median(t_err)),
            "r_err_deg_median": float(np.median(r_err)),
            "ransac_ms_median": float(np.median(ms)),
        }
    return out


def exp_landmark(rng: np.random.Generator, samples: int = 100000) -> dict:
    """XYZ vs inverse-depth model at KITTI ranges: depth bias and NEES of each Gaussian model.

    Args:
        rng: Generator.
        samples: Monte Carlo samples per depth.

    Returns:
        Results per depth.
    """
    sigma = 0.5
    out = {}
    for z in (10.0, 30.0, 60.0, 100.0):
        d0 = CAM.fx * CAM.baseline / z
        noise = rng.normal(0, sigma, (samples, 2))  # u_l, u_r noise.
        d = d0 + noise[:, 0] - noise[:, 1]
        keep = d > 0.5  # Valid matches only.
        d = d[keep]
        z_hat = CAM.fx * CAM.baseline / d
        var_z = triangulation_covariance(np.column_stack([np.full(d.size, CAM.cx), np.full(d.size, CAM.cy)]), d, CAM, sigma)[:, 2, 2]
        nees_xyz = (z_hat - z) ** 2 / var_z  # 1-D: should average 1.
        rho_hat, var_rho = 1 / z_hat, var_z / z_hat**4
        nees_rho = (rho_hat - 1 / z) ** 2 / var_rho
        m = 20  # Fuse m independent observations of the same point.
        dm = d0 + rng.normal(0, sigma, (20000, m)) - rng.normal(0, sigma, (20000, m))
        dm = np.where(dm > 0.5, dm, np.nan)
        zm = CAM.fx * CAM.baseline / dm
        w_xyz = dm**4  # 1 / var(Z) evaluated at each estimate (var Z ~ Z^4).
        z_fused_xyz = np.nansum(w_xyz * zm, axis=1) / np.nansum(w_xyz, axis=1)  # Information-form fusion in Z.
        z_fused_rho = 1.0 / np.nanmean(1.0 / zm, axis=1)  # Equal-variance fusion in rho.
        out[f"{z:.0f}"] = {
            "fused20_bias_xyz_pct": float(100 * (np.mean(z_fused_xyz) - z) / z),
            "fused20_bias_rho_pct": float(100 * (np.mean(z_fused_rho) - z) / z),
            "disparity_px": float(d0),
            "depth_bias_pct": float(100 * (np.median(z_hat) - z) / z),
            "mean_depth_bias_pct": float(100 * (np.mean(z_hat) - z) / z),
            "nees_xyz": float(np.mean(nees_xyz)),
            "nees_rho": float(np.mean(nees_rho)),
            "frac_xyz_gt_99": float(np.mean(nees_xyz > stats.chi2.ppf(0.99, 1))),
            "frac_rho_gt_99": float(np.mean(nees_rho > stats.chi2.ppf(0.99, 1))),
        }
    return out


def exp_inflation(rng: np.random.Generator, trials: int = 300, frames: int = 30) -> dict:
    """Fuse re-observations of a 25 m point from poses with unmodelled error; track anchor NEES.

    Args:
        rng: Generator.
        trials: Trials.
        frames: Re-observations.

    Returns:
        NEES vs number of fused observations, with and without inflation.
    """
    sigma = 0.5
    pose_cov = np.diag([0.01**2] * 3 + [np.radians(0.05) ** 2] * 3)  # 1 cm, 0.05 deg per frame (pose error of the triangulating camera).
    truth = np.array([3.0, -1.0, 25.0])  # World point (world = first camera).
    nees = {"none": np.zeros(frames), "inflated": np.zeros(frames)}
    for _ in range(trials):
        state = {}
        for k in range(frames):
            cam_true = make_pose(np.eye(3), np.array([0.0, 0.0, 0.3 * k]))  # T_world_cam (driving forward).
            err = rng.multivariate_normal(np.zeros(6), pose_cov) if k > 0 else np.zeros(6)
            cam_est = invert_pose(se3_exp(err) @ invert_pose(cam_true))  # Estimated pose with error (left perturbation of T_cw).
            p_c = (truth - cam_true[:3, 3]) @ cam_true[:3, :3]  # True camera point.
            u = CAM.fx * p_c[0] / p_c[2] + CAM.cx
            v = CAM.fy * p_c[1] / p_c[2] + CAM.cy
            n_l, n_r = rng.normal(0, sigma, 2)  # Left and right column noise (u_l enters u and the disparity).
            d = CAM.fx * CAM.baseline / p_c[2] + n_l - n_r
            u_n = u + n_l
            z = CAM.fx * CAM.baseline / d
            obs_c = np.array([(u_n - CAM.cx) * z / CAM.fx, (v + rng.normal(0, sigma) - CAM.cy) * z / CAM.fy, z])
            cov_c = triangulation_covariance(np.array([[u_n, v]]), np.array([d]), CAM, sigma)[0]
            obs_w = cam_est[:3, :3] @ obs_c + cam_est[:3, 3]
            cov_w = cam_est[:3, :3] @ cov_c @ cam_est[:3, :3].T
            for mode in nees:
                c = cov_w + (pose_uncertainty_covariance(obs_c[None], cam_est, pose_cov)[0] if mode == "inflated" and k > 0 else 0)
                if mode not in state:
                    state[mode] = (obs_w[None], c[None])
                else:
                    state[mode] = fuse_xyz(state[mode][0], state[mode][1], obs_w[None], c[None])
                e = state[mode][0][0] - truth
                nees[mode][k] += float(e @ np.linalg.solve(state[mode][1][0], e)) / trials
    return {m: v.tolist() for m, v in nees.items()}


def exp_drift(rng: np.random.Generator, frames: int = 40) -> dict:
    """Chained KLT vs birth-template re-alignment on a forward-moving view of a textured plane.

    Args:
        rng: Generator.
        frames: Frames.

    Returns:
        Median pixel error per frame for both trackers.
    """
    h, w = 376, 1241
    big = cv2.GaussianBlur(rng.uniform(0, 255, (h * 3, w * 3)).astype(np.float32), (0, 0), 2.5)
    big = cv2.normalize(big, None, 0, 255, cv2.NORM_MINMAX)
    depth0, step, lateral = 20.0, 0.25, 0.03  # Plane depth, forward and lateral motion per frame (m).
    kk = np.array([[CAM.fx, 0, w / 2], [0, CAM.fy, h / 2], [0, 0, 1.0]])

    def render(k: int) -> np.ndarray:
        """Image of the plane after k frames (homography of a fronto-parallel plane)."""
        z = depth0 - step * k
        s = depth0 / z  # Magnification.
        tx = -lateral * k * CAM.fx / z  # Shift.
        m = np.array([[s, 0, w / 2 - s * (w / 2 + w) + tx], [0, s, h / 2 - s * (h / 2 + h)]])  # Texture coords -> image.
        return cv2.warpAffine(big, m, (w, h), flags=cv2.INTER_CUBIC).astype(np.uint8)

    def true_position(p0: np.ndarray, k: int) -> np.ndarray:
        """True pixel of a birth pixel after k frames."""
        z = depth0 - step * k
        s = depth0 / z
        return np.column_stack([w / 2 + s * (p0[:, 0] - w / 2) - lateral * k * CAM.fx / z, h / 2 + s * (p0[:, 1] - h / 2)])

    images = [render(k) for k in range(frames)]
    corners = cv2.goodFeaturesToTrack(images[0], 400, 0.01, 10, mask=None).reshape(-1, 2)
    corners = corners[(corners[:, 0] > 300) & (corners[:, 0] < w - 300) & (corners[:, 1] > 60) & (corners[:, 1] < h - 60)]
    chained = corners.astype(np.float32).copy()
    aligned = corners.astype(np.float64).copy()
    alive = np.ones(len(corners), dtype=bool)
    err_klt, err_align = [0.0], [0.0]
    for k in range(1, frames):
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(images[k - 1], images[k], chained.reshape(-1, 1, 2), None, winSize=(21, 21), maxLevel=3)
        chained = nxt.reshape(-1, 2)
        alive &= st.ravel() == 1
        truth = true_position(corners, k)
        klt_now = chained.astype(np.float64)
        z = depth0 - step * k
        cur_from_birth = make_pose(np.eye(3), np.array([-lateral * k, 0.0, -step * k]))  # Camera moved forward/right.
        affine, _ = predict_affine(kk, corners.astype(np.float64), np.full(len(corners), depth0), cur_from_birth)
        new, ok = align_warped(images[0], images[k], corners.astype(np.float64), affine, klt_now, max_shift_px=3.0)
        aligned = np.where(ok[:, None], new, klt_now)
        keep = alive & (truth[:, 0] > 20) & (truth[:, 0] < w - 20) & (truth[:, 1] > 20) & (truth[:, 1] < h - 20)
        err_klt.append(float(np.median(np.linalg.norm(klt_now[keep] - truth[keep], axis=1))))
        err_align.append(float(np.median(np.linalg.norm(aligned[keep] - truth[keep], axis=1))))
        _ = z
    return {"klt": err_klt, "aligned": err_align, "tracks": int(alive.sum())}


def exp_extrinsic(rng: np.random.Generator, trials: int = 60) -> dict:
    """RQ3 preview: the true right camera is rotated away from calibration; the pipeline assumes the nominal rig.

    Landmarks are triangulated in the previous frame with the nominal (wrong) rig, then the current pose is
    estimated with the rigid-rig LM (u_l, v_l, u_r) or left-only LM (u_l, v_l). Also reports the median vertical
    disparity v_l - v_r, which the pipeline logs per frame.

    Args:
        rng: Generator.
        trials: Trials per error level.

    Returns:
        Per axis and error angle: translation error (% of the 0.9 m step), rotation error, vertical disparity.
    """
    out: dict = {}
    sigma = 0.5
    for axis, name in ((0, "pitch"), (1, "yaw"), (2, "roll")):
        rows = []
        for deg in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
            rvec = np.zeros(3)
            rvec[axis] = np.radians(deg)
            r_delta, _ = cv2.Rodrigues(rvec)
            right_from_left = make_pose(r_delta, np.array([-CAM.baseline, 0.0, 0.0]))  # True rig (rotated right camera).
            res = {"rigid_t": [], "rigid_r": [], "left_t": [], "left_r": [], "vdisp": [], "vmad": []}
            for _ in range(trials):
                pts_w, truth, _, _ = scene(rng, 300)  # Points in the previous camera = reference.

                def observe(points: np.ndarray, cam_from_ref: np.ndarray, rig: np.ndarray = right_from_left) -> tuple[np.ndarray, np.ndarray]:
                    """Left/right pixels through the TRUE rig, with noise."""
                    pl = points @ cam_from_ref[:3, :3].T + cam_from_ref[:3, 3]
                    pr = pl @ rig[:3, :3].T + rig[:3, 3]
                    left = np.column_stack([CAM.fx * pl[:, 0] / pl[:, 2] + CAM.cx, CAM.fy * pl[:, 1] / pl[:, 2] + CAM.cy])
                    right = np.column_stack([CAM.fx * pr[:, 0] / pr[:, 2] + CAM.cx, CAM.fy * pr[:, 1] / pr[:, 2] + CAM.cy])
                    return left + rng.normal(0, sigma, left.shape), right + rng.normal(0, sigma, right.shape)

                l0, r0 = observe(pts_w, np.eye(4))  # Previous frame.
                d = l0[:, 0] - r0[:, 0]
                ok = d > 1.0
                z = CAM.fx * CAM.baseline / d[ok]  # Nominal-rig triangulation.
                tri = np.column_stack([(l0[ok, 0] - CAM.cx) * z / CAM.fx, (l0[ok, 1] - CAM.cy) * z / CAM.fy, z])
                l1, r1 = observe(pts_w[ok], truth)  # Current frame.
                cov = triangulation_covariance(l0[ok], d[ok], CAM, sigma)
                cfg = LMConfig(pixel_sigma=sigma)
                for key, right in (("rigid", r1), ("left", None)):
                    lm = refine_pose_lm(truth, tri, l1, K, right, CAM.baseline, cov, cfg)
                    e = pose_error_vec(lm.camera_from_reference, truth)
                    res[f"{key}_t"].append(100 * np.linalg.norm(e[:3]) / 0.9)
                    res[f"{key}_r"].append(np.degrees(np.linalg.norm(e[3:])))
                vd = l1[:, 1] - r1[:, 1]  # Vertical disparity as logged by the pipeline.
                res["vdisp"].append(float(np.median(vd)))
                res["vmad"].append(float(1.4826 * np.median(np.abs(vd - np.median(vd)))))
            rows.append({"deg": deg, **{k: float(np.mean(v)) for k, v in res.items()}})
        out[name] = rows
    return out


def main() -> None:
    """Run all experiments and write figures + JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("report/validation"))
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    results = {
        "nees": exp_nees(rng),
        "ransac": exp_ransac(rng),
        "landmark": exp_landmark(rng),
        "inflation": exp_inflation(rng),
        "drift": exp_drift(rng),
        "extrinsic": exp_extrinsic(rng),
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.1))
    ax = axes[0]  # NEES histogram.
    bins = np.linspace(0, 30, 41)
    x = np.linspace(0, 30, 300)
    ax.hist(results["nees"]["shared_0.5"]["nees"], bins=bins, density=True, alpha=0.6, color="#2ca02c", label="shared $\\sigma$ (0.5 px)")
    ax.hist(results["nees"]["mismatched_1.0"]["nees"], bins=bins, density=True, alpha=0.6, color="#d62728", label="LM $\\sigma$ = 1 px")
    ax.plot(x, stats.chi2.pdf(x, 6), "k--", lw=1, label="$\\chi^2_6$")
    ax.set_xlabel("pose NEES")
    ax.set_title("(a) LM consistency")
    ax.legend(frameon=False)
    ax = axes[1]  # Inflation.
    n = np.arange(1, len(results["inflation"]["none"]) + 1)
    ax.plot(n, results["inflation"]["none"], color="#d62728", label="fused, no inflation")
    ax.plot(n, results["inflation"]["inflated"], color="#2ca02c", label="fused + pose inflation")
    ax.axhline(3, color="k", ls="--", lw=1, label="ideal (3)")
    ax.set_xlabel("fused observations")
    ax.set_ylabel("anchor NEES")
    ax.set_title("(b) Anchor over-confidence")
    ax.legend(frameon=False)
    ax = axes[2]  # Drift.
    f = np.arange(len(results["drift"]["klt"]))
    ax.plot(f, results["drift"]["klt"], color="#d62728", label="chained KLT")
    ax.plot(f, results["drift"]["aligned"], color="#2ca02c", label="birth-template re-alignment")
    ax.set_xlabel("frames since birth")
    ax.set_ylabel("median track error [px]")
    ax.set_title("(c) Track drift")
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "validation.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.0))
    colours = {"pitch": "#1f77b4", "yaw": "#d62728", "roll": "#9467bd"}
    for name, rows in results["extrinsic"].items():
        deg = [r["deg"] for r in rows]
        axes[0].plot(deg, [r["rigid_t"] for r in rows], "-o", ms=3, color=colours[name], label=f"rigid rig, {name} error")
        axes[0].plot(deg, [r["left_t"] for r in rows], "--", color=colours[name], label=f"left only, {name} error")
        axes[1].plot(deg, [abs(r["vdisp"]) for r in rows], "-o", ms=3, color=colours[name], label=f"{name}: |median|")
        axes[1].plot(deg, [r["vmad"] for r in rows], ":", color=colours[name], label=f"{name}: MAD")
    axes[0].set_xlabel("right-camera rotation error [deg]")
    axes[0].set_ylabel("frame translation error [%]")
    axes[0].set_yscale("log")
    axes[0].set_title("(a) Sensitivity to extrinsic rotation error")
    axes[0].legend(frameon=False, ncol=2, fontsize=6)
    axes[1].set_xlabel("right-camera rotation error [deg]")
    axes[1].set_ylabel("$v_l - v_r$ statistic [px]")
    axes[1].set_title("(b) Vertical-disparity signal")
    axes[1].legend(frameon=False, ncol=2, fontsize=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "extrinsic.pdf")
    for key in ("nees",):  # Drop raw samples from the JSON.
        for v in results[key].values():
            v.pop("nees")
    (out / "validation.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results["extrinsic"], indent=1))
    print("inflation final", {m: round(v[-1], 2) for m, v in results["inflation"].items()})


if __name__ == "__main__":
    main()
