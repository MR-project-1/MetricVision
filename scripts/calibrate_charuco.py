#!/usr/bin/env python3
"""Stereo calibration with a ChArUco board (partial views welcome).

  # frames extracted from a bag by scripts/extract_rosbag.py (left/ and right/ subfolders)
  python scripts/calibrate_charuco.py --frames /data/d455_board --board configs/charuco_board.yaml \
      --out /data/wheelchair/calib.yaml --factory /data/d455_board/factory_calib.yaml

  # check the whole tool on rendered images first (no hardware needed)
  python scripts/calibrate_charuco.py --selftest

Pipeline
  1. detect ChArUco corners in every left/right frame (IDs -> partial boards are fine)
  2. drop blurred frames, then pick --max-views frames that are maximally different in
     position, tilt and distance (greedy farthest-point sampling)
  3. per-camera Zhang initialisation, robust T_RL initialisation, then one joint LM over
     ALL observations of both cameras (not only corners seen by both), Huber first, then
     outlier / bad-view rejection and a final least-squares pass
     (--model auto tries none/radial2/radtan4/radtan5 and keeps the simplest model that
     is not worse on hold-out views)
  4. report: RMS per camera, 1-sigma uncertainty of every parameter (incl. baseline and
     extrinsic rotation), hold-out RMS, rectification check, coverage, tilt, comparison
     with the factory calibration, and a diagnostics figure

Outputs: calib.yaml (RawStereoCalib, used by the VO), <out>_report.json, <out>_diagnostics.png
"""

import argparse
import glob
import json
import os
import sys
import tempfile
import time

import cv2
import numpy as np

from metricvision.calibration._geometry import rot_angle  # noqa: E402
from metricvision.calibration.camera import RawStereoCalib  # noqa: E402
from metricvision.calibration.charuco import BoardSpec, CharucoDetector, approx_K, select_views, sharpness, view_features  # noqa: E402
from metricvision.calibration.solver import DIST_MODELS, StereoView, calibrate_stereo_views  # noqa: E402

EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif")


def list_pairs(left_dir, right_dir):
    lefts = sorted(sum((glob.glob(os.path.join(left_dir, e)) for e in EXTS), []))
    rights = {os.path.splitext(os.path.basename(p))[0]: p for p in sum((glob.glob(os.path.join(right_dir, e)) for e in EXTS), [])}
    return [(p, rights[os.path.splitext(os.path.basename(p))[0]]) for p in lefts if os.path.splitext(os.path.basename(p))[0] in rights]


def detect_all(pairs, det, stride, verbose=True):
    rows = []
    t0 = time.time()
    for k, (pl, pr) in enumerate(pairs[::stride]):
        L = cv2.imread(pl, cv2.IMREAD_GRAYSCALE)
        R = cv2.imread(pr, cv2.IMREAD_GRAYSCALE)
        dl, dr = det.detect(L), det.detect(R)
        rows.append(
            {
                "name": os.path.splitext(os.path.basename(pl))[0],
                "L": dl,
                "R": dr,
                "sharp": max(sharpness(L, dl.uv), sharpness(R, dr.uv)),
                "size": (L.shape[1], L.shape[0]),
                "paths": (pl, pr),
            }
        )
        if verbose and (k % 50 == 0):
            print(f"  detected {k + 1}/{len(pairs[::stride])} frames ({time.time() - t0:.0f} s)", flush=True)
    return rows


def choose(rows, spec, max_views, hfov, min_sharp_ratio=0.4):
    usable = [r for r in rows if r["L"].n or r["R"].n]
    if not usable:
        return [], {"detected": 0}
    med = np.median([r["sharp"] for r in usable])
    sharp = [r for r in usable if r["sharp"] >= min_sharp_ratio * med]
    size = usable[0]["size"]
    K = approx_K(size[0], size[1], hfov)
    feats, scores = [], []
    for r in sharp:
        d = r["L"] if r["L"].n >= r["R"].n else r["R"]
        feats.append(view_features(d, spec, K, size))
        scores.append(r["L"].n + r["R"].n + (spec.n_corners if (r["L"].n and r["R"].n) else 0))
    idx = select_views(feats, scores, max_views)
    stats = {
        "frames": len(rows),
        "detected": len(usable),
        "sharp": len(sharp),
        "selected": len(idx),
        "both_cameras_selected": sum(1 for i in idx if sharp[i]["L"].n and sharp[i]["R"].n),
    }
    return [sharp[i] for i in idx], stats


def compare_factory(calib, fac):
    out = {
        "baseline_ours_mm": 1000 * float(np.linalg.norm(calib.T)),
        "baseline_factory_mm": 1000 * float(np.linalg.norm(fac.T)),
        "rotation_diff_deg": float(np.degrees(rot_angle(calib.R @ fac.R.T))),
    }
    out["baseline_diff_mm"] = out["baseline_ours_mm"] - out["baseline_factory_mm"]
    for side, K1, K2 in (("left", calib.K_L, fac.K_L), ("right", calib.K_R, fac.K_R)):
        out[side] = {
            "dfx": float(K1[0, 0] - K2[0, 0]),
            "dfy": float(K1[1, 1] - K2[1, 1]),
            "dcx": float(K1[0, 2] - K2[0, 2]),
            "dcy": float(K1[1, 2] - K2[1, 2]),
        }
    return out


def plot_diagnostics(res, size, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a897f"
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    o = res.obs
    w, h = size
    fig, ax = plt.subplots(2, 2, figsize=(9, 6.2))
    for k, (cam, col, name) in enumerate(((0, BLUE, "Left"), (1, ORANGE, "Right"))):
        a = ax[0, k]
        m = o["cam"] == cam
        a.scatter(o["uv"][m & o["inlier"], 0], o["uv"][m & o["inlier"], 1], s=2, color=col, alpha=0.5, label="corners used")
        a.scatter(o["uv"][m & ~o["inlier"], 0], o["uv"][m & ~o["inlier"], 1], s=10, marker="x", color=GREY, label="rejected")
        a.set_xlim(0, w)
        a.set_ylim(h, 0)
        a.set_aspect("equal")
        cov = res.report["coverage_left" if cam == 0 else "coverage_right"]
        a.set_title(f"{name} image coverage ({100 * cov:.0f} % of 8x6 cells)", loc="left")
        a.legend(loc="lower right", markerscale=3)
    pv = res.report["per_view"]
    x = np.arange(len(pv))
    a = ax[1, 0]
    a.bar(x - 0.2, [p["rms_L"] for p in pv], 0.38, color=BLUE, label="left")
    a.bar(x + 0.2, [p["rms_R"] for p in pv], 0.38, color=ORANGE, label="right")
    a.axhline(0.5, color=GREY, lw=0.8, ls="--")
    a.set_xlabel("view")
    a.set_ylabel("RMS [px]")
    a.set_title("Reprojection RMS per view (dashed: 0.5 px)", loc="left")
    a.legend()
    a = ax[1, 1]
    for cam, col, name in ((0, BLUE, "left"), (1, ORANGE, "right")):
        m = (o["cam"] == cam) & o["inlier"]
        a.hist(o["residual_px"][m], bins=40, range=(0, 1.5), histtype="step", lw=1.6, color=col, label=name)
    a.set_xlabel("residual [px]")
    a.set_ylabel("corners")
    a.set_title("Residual distribution", loc="left")
    a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_detections(rows, out_dir, n=6):
    os.makedirs(out_dir, exist_ok=True)
    for r in rows[:: max(1, len(rows) // n)][:n]:
        imgs = []
        for side, p in zip(("L", "R"), r["paths"]):
            img = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
            d = r[side]
            for i, (u, v) in zip(d.ids, d.uv):
                cv2.circle(img, (int(round(u)), int(round(v))), 3, (0, 200, 0), -1)
                cv2.putText(img, str(i), (int(u) + 3, int(v) - 3), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 255), 1)
            imgs.append(img)
        cv2.imwrite(os.path.join(out_dir, f"{r['name']}.png"), np.hstack(imgs))


def run(a):
    spec = BoardSpec.from_yaml(a.board)
    if a.frames:
        a.left, a.right = os.path.join(a.frames, "left"), os.path.join(a.frames, "right")
    pairs = list_pairs(a.left, a.right)
    if not pairs:
        sys.exit(f"no matching image pairs in {a.left} / {a.right}")
    stride = a.stride or max(1, len(pairs) // 1500)
    print(f"{len(pairs)} stereo pairs; detecting every {stride}. frame ...")
    det = CharucoDetector(spec, min_corners=a.min_corners)
    rows = detect_all(pairs, det, stride)
    print(f"legacy layout: {det.legacy}")
    sel, st = choose(rows, spec, a.max_views, a.approx_fov)
    print(
        f"frames {st.get('frames', 0)}, board found {st['detected']}, sharp {st.get('sharp', 0)}, "
        f"selected {st.get('selected', 0)} ({st.get('both_cameras_selected', 0)} seen by both cameras)"
    )
    if len(sel) < 6:
        sys.exit("too few usable views - check the board YAML (sizes, dictionary) and the images")
    size = sel[0]["size"]
    views = [StereoView(r["name"], r["L"].ids, r["L"].uv, r["R"].ids, r["R"].uv) for r in sel]
    t0 = time.time()
    if a.model == "auto":
        # simplest distortion model whose hold-out RMS is within 5 % of the best one
        trials = {}
        for m in ("none", "radial2", "radtan4", "radtan5"):
            r_ = calibrate_stereo_views(views, spec.corner_points(), size, m, holdout=max(a.holdout, 0.2))
            trials[m] = r_
            print(f"  model {m:8s}: train {r_.report['rms_joint_px']:.3f} px  hold-out {r_.report['holdout']['rms_px']:.3f} px")
        best = min(t.report["holdout"]["rms_px"] for t in trials.values())
        a.model = next(m for m, t in trials.items() if t.report["holdout"]["rms_px"] <= 1.05 * best)
        print(f"  -> selected distortion model: {a.model}")
        res = trials[a.model]
        res.report["model_selection"] = {
            m: {"train_rms_px": t.report["rms_joint_px"], "holdout_rms_px": t.report["holdout"]["rms_px"]} for m, t in trials.items()
        }
    else:
        res = calibrate_stereo_views(views, spec.corner_points(), size, a.model, holdout=a.holdout, verbose=True)
    rep = res.report
    rep["selection"] = st
    rep["board"] = spec.to_dict()
    rep["solve_time_s"] = time.time() - t0
    if a.factory:
        rep["factory_comparison"] = compare_factory(res.calib, RawStereoCalib.from_yaml(a.factory))

    base = os.path.splitext(a.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    summary = {k: rep[k] for k in ("distortion_model", "rms_left_px", "rms_right_px", "rms_joint_px", "baseline_m", "n_views", "n_views_both")}
    summary["baseline_sigma_mm"] = rep["uncertainty_1sigma"]["baseline_mm"]
    summary["rotation_sigma_deg"] = rep["uncertainty_1sigma"]["rotation_deg"]
    res.calib.meta = {
        "calibration": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in summary.items()},
        "board": spec.to_dict(),
    }
    res.calib.to_yaml(a.out)
    with open(base + "_report.json", "w") as f:
        json.dump(rep, f, indent=2, default=float)
    plot_diagnostics(res, size, base + "_diagnostics.png")
    if a.save_detections:
        draw_detections(sel, base + "_detections")

    u = rep["uncertainty_1sigma"]
    print("\n================ stereo calibration ================")
    print(
        f"model {a.model}; {rep['n_views']} views ({rep['n_views_both']} both, {rep['n_views_left_only']} "
        f"left only, {rep['n_views_right_only']} right only); dropped {len(rep['dropped_views'])}; "
        f"{rep['n_outlier_observations']} outlier corners"
    )
    print(f"RMS   left {rep['rms_left_px']:.3f}  right {rep['rms_right_px']:.3f}  joint {rep['rms_joint_px']:.3f} px")
    if "holdout" in rep:
        print(
            f"hold-out RMS {rep['holdout']['rms_px']:.3f} px on {rep['holdout']['n_views']} unseen views "
            f"(x{rep['holdout']['ratio_to_train']:.2f} of training)"
        )
    c = res.calib
    for side, K, D, s in (("left ", c.K_L, c.D_L, "_L"), ("right", c.K_R, c.D_R, "_R")):
        print(
            f"{side} fx {K[0, 0]:.2f}±{u['fx' + s]:.2f}  fy {K[1, 1]:.2f}±{u['fy' + s]:.2f}  "
            f"cx {K[0, 2]:.2f}±{u['cx' + s]:.2f}  cy {K[1, 2]:.2f}±{u['cy' + s]:.2f}  D {np.round(D, 4)}"
        )
    print(
        f"baseline {1000 * rep['baseline_m']:.2f} ± {u['baseline_mm']:.3f} mm   "
        f"extrinsic rotation ± {u['rotation_deg']:.4f} deg (1 sigma, assumes the measured square size)"
    )
    rc = rep["rectification_check"]
    if rc.get("n"):
        print(f"rectified |v_L - v_R|: mean {rc['mean_abs_px']:.3f}  p95 {rc['p95_abs_px']:.3f} px")
    print(
        f"coverage L {100 * rep['coverage_left']:.0f} %  R {100 * rep['coverage_right']:.0f} %   "
        f"tilt {rep['tilt_deg']['min']:.0f}-{rep['tilt_deg']['max']:.0f} deg (median "
        f"{rep['tilt_deg']['median']:.0f})   distance {rep['distance_m']['min']:.2f}-{rep['distance_m']['max']:.2f} m"
    )
    if "factory_comparison" in rep:
        fc = rep["factory_comparison"]
        print(
            f"vs factory: baseline {fc['baseline_diff_mm']:+.2f} mm, rotation {fc['rotation_diff_deg']:.3f} deg, "
            f"left dfx {fc['left']['dfx']:+.2f} dcx {fc['left']['dcx']:+.2f} dcy {fc['left']['dcy']:+.2f} px"
        )
    for w_ in rep["warnings"]:
        print("WARNING:", w_)
    print(f"\nwritten {a.out}, {base}_report.json, {base}_diagnostics.png")
    return res


def selftest():
    """Render a known rig + board, run the full CLI path, check the result."""
    from metricvision.calibration.charuco_synth import CharucoRenderer, default_rig, random_board_pose

    tmp = tempfile.mkdtemp(prefix="charuco_selftest_")
    spec = BoardSpec()
    with open(os.path.join(tmp, "board.yaml"), "w") as f:
        import yaml

        yaml.safe_dump({"board": spec.to_dict()}, f)
    KL, DL, KR, DR, T_RL, size = default_rig()
    rend = CharucoRenderer(spec)
    rng = np.random.default_rng(3)
    os.makedirs(os.path.join(tmp, "left"))
    os.makedirs(os.path.join(tmp, "right"))
    print(f"rendering 60 synthetic stereo frames into {tmp} ...")
    for i in range(60):
        T = random_board_pose(rng, spec)
        cv2.imwrite(os.path.join(tmp, "left", f"{i:04d}.png"), rend.render(KL, DL, T, size, rng=rng))
        cv2.imwrite(os.path.join(tmp, "right", f"{i:04d}.png"), rend.render(KR, DR, T_RL @ T, size, rng=rng))
    a = argparse.Namespace(
        frames=tmp,
        left=None,
        right=None,
        board=os.path.join(tmp, "board.yaml"),
        out=os.path.join(tmp, "calib.yaml"),
        factory=None,
        model="radtan4",
        max_views=40,
        min_corners=10,
        stride=1,
        holdout=0.2,
        approx_fov=70.0,
        save_detections=True,
    )
    res = run(a)
    c = res.calib
    e_b = 1000 * abs(np.linalg.norm(c.T) - np.linalg.norm(T_RL[:3, 3]))
    e_r = np.degrees(rot_angle(c.R @ T_RL[:3, :3].T))
    e_f = max(abs(c.K_L[0, 0] - KL[0, 0]), abs(c.K_R[0, 0] - KR[0, 0]))
    ok = e_b < 0.3 and e_r < 0.1 and e_f < 2.0
    print(f"\nSELFTEST {'PASSED' if ok else 'FAILED'}: baseline error {e_b:.3f} mm, rotation error {e_r:.4f} deg, focal error {e_f:.2f} px")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", help="folder with left/ and right/ (as written by extract_rosbag.py)")
    ap.add_argument("--left")
    ap.add_argument("--right")
    ap.add_argument("--board", default=os.path.join(os.path.dirname(__file__), "..", "configs", "charuco_board.yaml"))
    ap.add_argument("--out", default="calib.yaml")
    ap.add_argument("--factory", help="factory calibration YAML for comparison (extract_rosbag.py writes one)")
    ap.add_argument(
        "--model",
        default="auto",
        choices=["auto"] + list(DIST_MODELS),
        help="distortion model; 'auto' picks the simplest one that does not lose on hold-out "
        "views ('none' is expected for the RealSense *_rect_raw streams)",
    )
    ap.add_argument("--max-views", type=int, default=40)
    ap.add_argument("--min-corners", type=int, default=10)
    ap.add_argument("--stride", type=int, default=0, help="use every N-th frame (default: auto)")
    ap.add_argument("--holdout", type=float, default=0.2, help="fraction of views kept out for validation")
    ap.add_argument("--approx-fov", type=float, default=87.0, help="horizontal FOV guess for view selection")
    ap.add_argument("--save-detections", action="store_true", help="save corner overlays for a few views")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not (a.frames or (a.left and a.right)):
        ap.error("give --frames DIR or --left DIR --right DIR")
    run(a)


if __name__ == "__main__":
    main()
