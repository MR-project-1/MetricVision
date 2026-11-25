"""Watch the stereo VO work on a wheelchair bag: keypoints, matches, PnP inliers and the growing trajectory.

Opens a live window (Tk; works with opencv-python-headless) and/or writes an MP4. The pipeline, calibration
and rectification are exactly those of the offline runs and the ROS 2 node (same YAML config).

Keys in the window: space = pause/resume, n = one step while paused, q / Esc = quit.

Usage:
    uv run python scripts/visualize_vo.py --bag ros_bags/run_01                       # live window, sparse
    uv run python scripts/visualize_vo.py --bag ros_bags/run_01 --pipeline dense
    uv run python scripts/visualize_vo.py --bag ros_bags/small_run_02 --window 5 --video results/vo_small_run_02.mp4 --no-show
"""

from __future__ import annotations

import argparse
import time
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

from metricvision.config import ExperimentConfig
from metricvision.datasets import DATASETS
from metricvision.pipeline import build_odometry
from metricvision.viz import VOVisualizer


def main() -> None:
    """Run the pipeline frame by frame and show / record the debug canvas."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/wheelchair.yaml"))
    parser.add_argument("--bag", type=str, default=None, help="Bag folder (overrides dataset.params.root).")
    parser.add_argument("--pipeline", choices=["sparse", "dense"], default="sparse")
    parser.add_argument("--window", type=int, default=None, help="Sparse windowed-BA size (0 = off, 5 = the BA runs).")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--video", type=Path, default=None, help="Write the canvas to this MP4.")
    parser.add_argument("--no-show", action="store_true", help="No window (only --video).")
    parser.add_argument("--scale", type=float, default=0.8, help="Window scale (the video is full size).")
    parser.add_argument("--realtime", action="store_true", help="Throttle to the bag's frame rate instead of as fast as possible.")
    args = parser.parse_args()
    if args.no_show and args.video is None:
        parser.error("--no-show needs --video")

    config = ExperimentConfig.from_yaml(args.config).with_overrides(dataset_root=args.bag, pipeline=args.pipeline, window=args.window)
    dataset = DATASETS.create(config.dataset.name, **config.dataset.params)
    odometry = build_odometry(config, dataset.camera)
    odometry.reset()
    reference = dataset.ground_truth()
    n = len(dataset) if args.max_frames is None else min(len(dataset), args.max_frames)
    window_note = f", BA window {args.window}" if args.window else ""
    viz = VOVisualizer(dataset.body_from_camera, f"{dataset.tag}  |  {args.pipeline}{window_note}  |  {n} frames")

    writer: cv2.VideoWriter | None = None
    state = {"index": 0, "paused": False, "step": False, "fps": 0.0, "last": time.perf_counter(), "t0": None, "wall0": None}

    def step() -> np.ndarray | None:
        """Process one frame; returns the canvas, or None at the end."""
        nonlocal writer
        k = state["index"]
        if k >= n:
            return None
        frame = dataset.load_frame(k)
        if args.realtime:  # Sleep until the frame's bag time.
            if state["t0"] is None:
                state["t0"], state["wall0"] = frame.timestamp, time.perf_counter()
            wait = (frame.timestamp - state["t0"]) - (time.perf_counter() - state["wall0"])
            if wait > 0:
                time.sleep(wait)
        result = odometry.process(frame)
        now = time.perf_counter()
        state["fps"] = 0.9 * state["fps"] + 0.1 / max(now - state["last"], 1e-6) if k else 0.0
        state["last"] = now
        ref = reference.poses[k] if reference is not None else None
        canvas = viz.render(k, frame.left, frame.right, result, ref, state["fps"])
        if args.video is not None:
            if writer is None:
                args.video.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (canvas.shape[1], canvas.shape[0]))
            writer.write(canvas)
        state["index"] = k + 1
        return canvas

    if args.no_show:
        for _ in range(n):
            step()
            if state["index"] % 50 == 0:
                print(f"\r{state['index']}/{n}", end="", flush=True)
        print()
    else:
        root = tk.Tk()
        root.title(f"MetricVision VO - {dataset.tag} ({args.pipeline})")
        label = tk.Label(root, bg="black")
        label.pack()
        photo: list[ImageTk.PhotoImage] = []  # Keep a reference or Tk drops the image.

        def on_key(event: tk.Event) -> None:
            if event.keysym in ("q", "Escape"):
                root.destroy()
            elif event.keysym == "space":
                state["paused"] = not state["paused"]
                state["last"] = time.perf_counter()
                state["t0"] = None
            elif event.keysym == "n":
                state["step"] = True

        root.bind("<Key>", on_key)

        def tick() -> None:
            if not state["paused"] or state["step"]:
                state["step"] = False
                canvas = step()
                if canvas is None:
                    root.title(root.title() + "  [finished - q to close]")
                    return
                if args.scale != 1.0:
                    canvas = cv2.resize(canvas, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
                photo[:] = [ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))]
                label.configure(image=photo[0])
            root.after(1, tick)

        root.after(1, tick)
        root.mainloop()

    if writer is not None:
        writer.release()
        print(f"wrote {args.video} ({state['index']} frames)")


if __name__ == "__main__":
    main()
