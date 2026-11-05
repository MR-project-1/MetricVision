"""Convert an MC-Calib / Kalibr stereo calibration into MetricVision's calib.yaml, verified on real images.

Usage (MC-Calib, recommended with an image check that resolves the pose convention and left/right automatically):
    uv run python scripts/import_calibration.py --input calibrated_cameras_data.yml \
        --check-folder ~/datasets/wheelchair/lab_01 --out ~/datasets/wheelchair/calib.yaml \
        --compare ~/datasets/wheelchair/lab_01/factory_calib.yaml
Kalibr:
    uv run python scripts/import_calibration.py --input camchain.yaml --out calib.yaml

The image check rectifies a few stereo pairs with every hypothesis (MC-Calib pose convention x left/right order),
tracks corners left -> right and keeps the hypothesis with positive disparities and the smallest rectified vertical
error (median |v_L - v_R|). The chosen calibration, the check table and the comparison with the factory calibration
(baseline / pitch / yaw / roll differences: the extrinsic errors studied in RQ3) are printed and stored in the YAML.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from metricvision.calibration.camera import RawStereoCalib, Rectifier
from metricvision.calibration.importers import compare_calibrations, detect_format, load_kalibr, load_mc_calib


def rectification_check(calib: RawStereoCalib, pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    """Rectify image pairs and measure the stereo geometry.

    Args:
        calib: Hypothesis.
        pairs: (left, right) grayscale images.

    Returns:
        Median |v_L - v_R| (px), 95th percentile, fraction of positive disparities, tracked corners, baseline (m).
    """
    try:
        rect = Rectifier(calib)
    except cv2.error:
        return {"median_dv": np.inf, "p95_dv": np.inf, "positive_disparity": 0.0, "n": 0, "baseline": float("nan")}
    dv, disp = [], []
    for left, right in pairs:
        lr, rr = rect(left, right)
        corners = cv2.goodFeaturesToTrack(lr, 800, 0.01, 8)
        if corners is None:
            continue
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(lr, rr, corners, None, winSize=(21, 21), maxLevel=4)
        back, status_b, _ = cv2.calcOpticalFlowPyrLK(rr, lr, tracked, None, winSize=(21, 21), maxLevel=4)
        ok = (status.ravel() == 1) & (status_b.ravel() == 1) & (np.linalg.norm((back - corners).reshape(-1, 2), axis=1) < 0.5)
        c, t = corners.reshape(-1, 2)[ok], tracked.reshape(-1, 2)[ok]
        dv.append(np.abs(c[:, 1] - t[:, 1]))
        disp.append(c[:, 0] - t[:, 0])
    if not dv:
        return {"median_dv": np.inf, "p95_dv": np.inf, "positive_disparity": 0.0, "n": 0, "baseline": rect.camera.baseline}
    dv_all, disp_all = np.concatenate(dv), np.concatenate(disp)
    return {
        "median_dv": float(np.median(dv_all)),
        "p95_dv": float(np.percentile(dv_all, 95)),
        "positive_disparity": float(np.mean(disp_all > 0.0)),
        "n": int(dv_all.size),
        "baseline": float(rect.camera.baseline),
    }


def _pairs(folder: Path, count: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Evenly spaced stereo pairs of a stereo folder."""
    lefts = sorted((folder / "left").glob("*.png"))
    rights = {p.name: p for p in (folder / "right").glob("*.png")}
    names = [p for p in lefts if p.name in rights]
    picks = names[:: max(1, len(names) // count)][:count]
    return [(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), cv2.imread(str(rights[p.name]), cv2.IMREAD_GRAYSCALE)) for p in picks]


def main() -> None:
    """Import, verify and write."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True, help="MC-Calib calibrated_cameras_data.yml or Kalibr camchain.yaml")
    parser.add_argument("--out", type=Path, required=True, help="output calib.yaml (RawStereoCalib)")
    parser.add_argument("--left", default=None, help="left camera (MC-Calib index, Kalibr key); default 0 / cam0")
    parser.add_argument("--right", default=None, help="right camera (MC-Calib index, Kalibr key); default 1 / cam1")
    parser.add_argument(
        "--pose-convention", choices=["auto", "cam_to_ref", "ref_to_cam"], default="auto", help="MC-Calib camera_pose_matrix direction"
    )
    parser.add_argument("--check-folder", type=Path, help="stereo folder (left/, right/) used to verify the calibration")
    parser.add_argument("--check-pairs", type=int, default=5)
    parser.add_argument("--compare", type=Path, help="another calibration (e.g. factory_calib.yaml) to compare with")
    args = parser.parse_args()
    fmt = detect_format(args.input)
    pairs = _pairs(args.check_folder, args.check_pairs) if args.check_folder else []
    if fmt == "mc_calib":
        left, right = int(args.left or 0), int(args.right or 1)
        conventions = ["cam_to_ref", "ref_to_cam"] if args.pose_convention == "auto" else [args.pose_convention]
        orders = [(left, right), (right, left)] if pairs else [(left, right)]
        hypotheses = [(conv, order, load_mc_calib(args.input, order[0], order[1], conv)) for conv in conventions for order in orders]
        if not pairs and args.pose_convention == "auto":
            print("WARNING: --pose-convention auto without --check-folder: assuming cam_to_ref. Verify with --check-folder.")
            hypotheses = hypotheses[:1]
        results = []
        for conv, order, calib in hypotheses:
            check = rectification_check(calib, pairs) if pairs else {}
            results.append((conv, order, calib, check))
            if check:
                print(
                    f"{conv:10s} left={order[0]} right={order[1]}: median |dv| {check['median_dv']:.3f} px, p95 {check['p95_dv']:.3f} px, "
                    f"positive disparity {100 * check['positive_disparity']:.0f} %, baseline {check['baseline']:.4f} m, {check['n']} corners"
                )
        if pairs:
            valid = [r for r in results if r[3]["positive_disparity"] > 0.9 and r[3]["baseline"] > 0]
            pool = valid or results
            conv, order, calib, check = min(pool, key=lambda r: r[3]["median_dv"])
            print(f"selected: {conv}, left={order[0]}, right={order[1]}")
            calib.meta.update({"check_median_dv_px": check["median_dv"], "check_p95_dv_px": check["p95_dv"]})
            if check["median_dv"] > 0.5:
                print("WARNING: rectified vertical error > 0.5 px: calibration and images may not belong together.")
        else:
            calib = results[0][2]
    elif fmt == "kalibr":
        calib = load_kalibr(args.input, args.left or "cam0", args.right or "cam1")
        if pairs:
            check = rectification_check(calib, pairs)
            print(f"check: median |dv| {check['median_dv']:.3f} px, positive disparity {100 * check['positive_disparity']:.0f} %")
            calib.meta.update({"check_median_dv_px": check["median_dv"], "check_p95_dv_px": check["p95_dv"]})
    else:
        calib = RawStereoCalib.from_yaml(str(args.input))
    print(f"baseline {1000 * np.linalg.norm(calib.T):.2f} mm, rotation {np.degrees(np.linalg.norm(cv2.Rodrigues(calib.R)[0])):.3f} deg")
    if args.compare:
        diff = compare_calibrations(calib, RawStereoCalib.from_yaml(str(args.compare)))
        print("vs " + args.compare.name + ": " + ", ".join(f"{k} {v:+.3f}" for k, v in diff.items()))
        calib.meta["comparison_" + args.compare.stem] = {k: round(v, 5) for k, v in diff.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    calib.to_yaml(str(args.out))
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
