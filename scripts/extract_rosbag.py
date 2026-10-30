#!/usr/bin/env python3
"""Extract a RealSense D455 ROS1/ROS2 bag into the stereo-folder layout.

  uv sync --extra rosbag   (or: pip install rosbags)
  python scripts/extract_rosbag.py --bag wheelchair_lab_01.bag --out data/wheelchair/lab_01 \
      [--left /camera/infra1/image_rect_raw --right /camera/infra2/image_rect_raw] \
      [--imu /camera/imu] [--calib data/wheelchair/calib.yaml]

Writes left/<ns>.png, right/<ns>.png, imu.csv (logged now for the later VIO stage)
and copies calib.yaml if given. Record with the IR emitter OFF
(`ros2 param set /camera/camera depth_module.emitter_enabled 0`, or
`emitter_enabled:=0` at launch), otherwise the projected dots are tracked as features.

If the bag also contains /camera/infra{1,2}/camera_info, the factory calibration is written
to factory_calib.yaml (use it with calibrate_charuco.py --factory).

Tested on synthetic ROS1 and ROS2 bags (tests/test_rosbag.py), not yet on a real D455 bag.

Note: the D455 publishes the IR streams rectified at the factory ('image_rect_raw').
If you calibrate yourself with scripts/calibrate_stereo.py, write calib.yaml with zero
distortion only if you calibrated on these same rectified streams.
"""

import argparse
import os
import shutil

import cv2
import numpy as np


def to_gray(msg):
    h, w, enc = msg.height, msg.width, msg.encoding
    buf = np.frombuffer(msg.data, np.uint8)
    if enc in ("mono8", "8UC1"):
        return buf.reshape(h, msg.step)[:, :w]
    if enc in ("mono16", "16UC1"):
        return (buf.view(np.uint16).reshape(h, msg.step // 2)[:, :w] >> 8).astype(np.uint8)
    if enc in ("bgr8", "rgb8"):
        img = buf.reshape(h, msg.step)[:, : 3 * w].reshape(h, w, 3)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY if enc == "bgr8" else cv2.COLOR_RGB2GRAY)
    raise ValueError(f"unsupported encoding {enc}")


def _f(msg, name):
    """CameraInfo field in ROS2 (lower case) or ROS1 (upper case) naming."""
    return np.asarray(getattr(msg, name, None) if hasattr(msg, name) else getattr(msg, name.upper()), float)


def write_factory_calib(info_l, info_r, path):
    """Factory stereo calibration from the two camera_info messages (RealSense convention:
    the right projection matrix carries P[0,3] = -fx * baseline, rectified streams have R = I)."""
    from metricvision.calibration.camera import RawStereoCalib

    KL, KR = _f(info_l, "k").reshape(3, 3), _f(info_r, "k").reshape(3, 3)
    PR = _f(info_r, "p").reshape(3, 4)
    PL = _f(info_l, "p").reshape(3, 4)
    b = -(PR[0, 3] - PL[0, 3]) / PR[0, 0]
    pad = lambda d: np.pad(d.ravel()[:5], (0, max(0, 5 - d.size)))
    RawStereoCalib(
        KL,
        pad(_f(info_l, "d")),
        KR,
        pad(_f(info_r, "d")),
        np.eye(3),
        np.array([-b, 0.0, 0.0]),
        int(info_l.width),
        int(info_l.height),
        {"source": "camera_info (factory)"},
    ).to_yaml(path)


def stamp_ns(msg):
    s = msg.header.stamp
    return int(s.sec) * 1_000_000_000 + int(s.nanosec)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--left", default="/camera/infra1/image_rect_raw")
    ap.add_argument("--right", default="/camera/infra2/image_rect_raw")
    ap.add_argument("--imu", default="/camera/imu", help="united IMU topic (unite_imu_method:=2)")
    ap.add_argument("--gyro", default="/camera/gyro/sample", help="used when the united topic is absent")
    ap.add_argument("--accel", default="/camera/accel/sample")
    ap.add_argument("--calib", default=None)
    ap.add_argument("--left-info", default="/camera/infra1/camera_info")
    ap.add_argument("--right-info", default="/camera/infra2/camera_info")
    ap.add_argument("--max-dt-ms", type=float, default=2.0, help="L/R pairing tolerance")
    a = ap.parse_args(argv)
    from pathlib import Path

    from rosbags.highlevel import AnyReader  # imported here so the rest of the repo has no dependency
    from rosbags.typesys import Stores, get_typestore

    os.makedirs(os.path.join(a.out, "left"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "right"), exist_ok=True)
    # stream: hold only unmatched images (a few frames), write each L/R pair as soon as it is
    # complete - a 3-min 848x480 bag would otherwise need several GB of RAM
    pending = {0: {}, 1: {}}
    n = 0
    tol = a.max_dt_ms * 1e6

    def push(side, t, img):
        nonlocal n
        other = pending[1 - side]
        match = min(other, key=lambda s_: abs(s_ - t)) if other else None
        if match is not None and abs(match - t) <= tol:
            L, R = (img, other.pop(match)) if side == 0 else (other.pop(match), img)
            tl = t if side == 0 else match
            cv2.imwrite(os.path.join(a.out, "left", f"{tl}.png"), L)
            cv2.imwrite(os.path.join(a.out, "right", f"{tl}.png"), R)
            n += 1
        else:
            pending[side][t] = img
        for d in pending.values():  # drop stale unmatched frames (> 1 s old)
            for s_ in [s_ for s_ in d if s_ < t - 1_000_000_000]:
                del d[s_]

    imu_rows, gyro_rows, accel_rows = [], [], []
    with AnyReader([Path(a.bag)], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        topics = {c.topic for c in reader.connections}
        missing = [t for t in (a.left, a.right) if t not in topics]
        if missing:
            raise SystemExit(f"topics {missing} not in bag; available: {sorted(topics)}")
        conns = [c for c in reader.connections if c.topic in (a.left, a.right, a.imu, a.gyro, a.accel, a.left_info, a.right_info)]
        infos = {}
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == a.imu:
                g, acc = msg.angular_velocity, msg.linear_acceleration
                imu_rows.append((stamp_ns(msg), g.x, g.y, g.z, acc.x, acc.y, acc.z))
                continue
            if conn.topic in (a.left_info, a.right_info):
                infos.setdefault(conn.topic, msg)
                continue
            if conn.topic == a.gyro:
                g = msg.angular_velocity
                gyro_rows.append((stamp_ns(msg), g.x, g.y, g.z))
                continue
            if conn.topic == a.accel:
                acc = msg.linear_acceleration
                accel_rows.append((stamp_ns(msg), acc.x, acc.y, acc.z))
                continue
            push(0 if conn.topic == a.left else 1, stamp_ns(msg), to_gray(msg))

    if imu_rows:
        np.savetxt(
            os.path.join(a.out, "imu.csv"), np.array(imu_rows), delimiter=",", header="t_ns,gx,gy,gz,ax,ay,az", comments="", fmt=["%d"] + ["%.9f"] * 6
        )
    for name, rows, hdr in (("gyro.csv", gyro_rows, "t_ns,gx,gy,gz"), ("accel.csv", accel_rows, "t_ns,ax,ay,az")):
        if rows:
            np.savetxt(os.path.join(a.out, name), np.array(rows), delimiter=",", header=hdr, comments="", fmt=["%d"] + ["%.9f"] * 3)
    if a.left_info in infos and a.right_info in infos:
        write_factory_calib(infos[a.left_info], infos[a.right_info], os.path.join(a.out, "factory_calib.yaml"))
        print(f"factory calibration from camera_info -> {os.path.join(a.out, 'factory_calib.yaml')}")
    if a.calib:
        shutil.copy(a.calib, os.path.join(a.out, "calib.yaml"))
    print(f"wrote {n} stereo pairs, {len(imu_rows)} IMU, {len(gyro_rows)} gyro, {len(accel_rows)} accel samples to {a.out}")


if __name__ == "__main__":
    main()
