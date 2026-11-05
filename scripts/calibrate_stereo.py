#!/usr/bin/env python3
"""Stereo calibration from checkerboard image pairs (Section 4.1).

  python scripts/calibrate_stereo.py --left calib/left --right calib/right \
      --cols 9 --rows 6 --square 0.025 --out data/wheelchair/calib.yaml \
      [--factory factory_d455.yaml]

Left/right images are paired by file name. Zhang initialisation (OpenCV), then our
own joint LM over Eq. (2). Prints RMS reprojection error (target < 0.5 px), the
baseline, and - if a factory calibration is given - the difference to it.
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

from metricvision.calibration._geometry import rot_angle  # noqa: E402
from metricvision.calibration.camera import RawStereoCalib  # noqa: E402
from metricvision.calibration.solver import board_points, calibrate_stereo, find_corners  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--cols", type=int, required=True, help="inner corners per row")
    ap.add_argument("--rows", type=int, required=True, help="inner corners per column")
    ap.add_argument("--square", type=float, required=True, help="square size in metres")
    ap.add_argument("--out", required=True)
    ap.add_argument("--factory", default=None, help="factory calibration YAML for comparison")
    a = ap.parse_args()

    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    lefts = sorted(sum((glob.glob(os.path.join(a.left, e)) for e in exts), []))
    obj = board_points(a.cols, a.rows, a.square)
    pL, pR, size = [], [], None
    for fl in lefts:
        fr = os.path.join(a.right, os.path.basename(fl))
        if not os.path.exists(fr):
            continue
        L = cv2.imread(fl, cv2.IMREAD_GRAYSCALE)
        R = cv2.imread(fr, cv2.IMREAD_GRAYSCALE)
        size = (L.shape[1], L.shape[0])
        cl, cr = find_corners(L, a.cols, a.rows), find_corners(R, a.cols, a.rows)
        if cl is not None and cr is not None:
            pL.append(cl)
            pR.append(cr)
    print(f"board found in {len(pL)}/{len(lefts)} pairs")
    if len(pL) < 8:
        sys.exit("need at least 8 good pairs (aim for 25-40 with varied tilt and distance)")

    calib, rep = calibrate_stereo(obj, pL, pR, size)
    calib.meta = {"calibration_report": rep}
    calib.to_yaml(a.out)
    print(json.dumps(rep, indent=2))
    ok = rep["rms_joint_px"] < 0.5
    print(f"joint RMS = {rep['rms_joint_px']:.3f} px  ({'meets' if ok else 'ABOVE'} the 0.5 px target)")
    print(f"baseline  = {1000 * rep['baseline_m']:.2f} mm")

    if a.factory:
        f = RawStereoCalib.from_yaml(a.factory)
        print("\n-- comparison with factory calibration --")
        print(f"baseline   ours {1000 * np.linalg.norm(calib.T):.2f} mm   factory {1000 * np.linalg.norm(f.T):.2f} mm")
        print(f"R_RL diff  {np.degrees(rot_angle(calib.R @ f.R.T)):.3f} deg")
        for side, K1, K2 in (("left", calib.K_L, f.K_L), ("right", calib.K_R, f.K_R)):
            d = np.array([K1[0, 0] - K2[0, 0], K1[1, 1] - K2[1, 1], K1[0, 2] - K2[0, 2], K1[1, 2] - K2[1, 2]])
            print(f"{side:5s} dfx={d[0]:+.2f} dfy={d[1]:+.2f} dcx={d[2]:+.2f} dcy={d[3]:+.2f} px")
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
