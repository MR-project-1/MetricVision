"""Live comparison graph next to rviz2: wheel odometry, lidar odometry, classical VO and learned VO, with save buttons.

A small window (matplotlib in Tk) that subscribes to the anchored paths published by ``vo_node`` and redraws twice a
second:

* top: the four paths from above, all starting at the first VO frame (same frame and colours as the report),
* bottom: how far each VO path is from the reference (lidar if it publishes, else wheels) at the latest pose, over time.

Buttons:

* **Save graph** writes this figure as ``plots/live_<time>.png`` in the run folder (``output_dir``),
* **Save full report** calls ``/vo/save``: the node writes data, plots, video so far and ``report.html`` without stopping.

Run through the launch file (``plot:=true``, the default) or ``ros2 run metricvision_ros vo_plot``.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from nav_msgs.msg import Path as PathMsg
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_srvs.srv import Trigger

SERIES = {  # topic: (label, colour, line width) - same colours as report.html and rviz2.
    "/vo/wheel_path": ("wheel odometry /odom", "#1a9b55", 2.6),
    "/vo/lidar_path": ("lidar odometry /Odometry", "#e07b00", 2.6),
    "/vo/path": ("stereo VO classical", "#1f7ae0", 1.6),
    "/vo_learned/path": ("stereo VO learned", "#8e44ad", 1.6),
}


def _length(xy: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))) if len(xy) > 1 else 0.0


def valid_part(data: np.ndarray, max_speed_mps: float = 3.0) -> tuple[np.ndarray, float | None]:
    """The path up to the first physically impossible jump (a wheelchair moves < 3 m/s), and when that was.

    Lidar odometry can diverge (DLIO once ran off by hundreds of metres); drawing that would hide every other path.
    """
    if len(data) < 2:
        return data, None
    dt = np.maximum(np.diff(data[:, 2]), 0.05)  # Path poses are thinned, so compare over at least 50 ms.
    speed = np.linalg.norm(np.diff(data[:, :2], axis=0), axis=1) / dt
    bad = np.nonzero(speed > max_speed_mps)[0]
    return (data, None) if len(bad) == 0 else (data[: bad[0] + 1], float(data[bad[0] + 1, 2]))


def draw(ax_map: Any, ax_err: Any, paths: dict[str, np.ndarray], history: dict[str, list[tuple[float, float]]]) -> None:
    """Draw the comparison into two matplotlib axes (used by the window; testable without a display)."""
    ax_map.clear()
    t0 = min((float(d[0, 2]) for d in paths.values() if len(d)), default=0.0)
    for topic, (label, colour, width) in SERIES.items():
        data = paths.get(topic)
        if data is None or not len(data):
            continue
        data, broke = valid_part(data)
        note = f", diverged at {broke - t0:.0f} s: cut" if broke is not None else ""
        ax_map.plot(data[:, 0], data[:, 1], color=colour, lw=width, label=f"{label} ({_length(data[:, :2]):.2f} m{note})")
        ax_map.plot(data[-1, 0], data[-1, 1], "o", color=colour, ms=6)
    ax_map.plot(0, 0, "k+", ms=10)
    ax_map.set_title("Top-down, all paths start at the first VO frame")
    ax_map.set_xlabel("x [m]")
    ax_map.set_ylabel("y [m]")
    ax_map.axis("equal")
    ax_map.grid(alpha=0.3)
    if paths:
        ax_map.legend(fontsize=8, loc="best")
    else:
        ax_map.text(0.5, 0.5, "waiting for /vo/path ...", transform=ax_map.transAxes, ha="center")
    ax_err.clear()
    for topic in ("/vo/path", "/vo_learned/path"):
        if history.get(topic):
            h = np.array(history[topic])
            ax_err.plot(h[:, 0], h[:, 1], color=SERIES[topic][1], label=SERIES[topic][0])
    ax_err.set_title("Distance of the latest VO pose to the reference (lidar, else wheels)", fontsize=9)
    ax_err.set_xlabel("time [s]")
    ax_err.set_ylabel("[m]")
    ax_err.grid(alpha=0.3)


class VOPlotNode(Node):
    """Collects the paths; the Tk thread draws them."""

    def __init__(self) -> None:
        super().__init__("metricvision_vo_plot")
        self.output_dir = str(self.declare_parameter("output_dir", "").value)
        self.lock = threading.Lock()
        self.paths: dict[str, np.ndarray] = {}  # topic -> (N, 3) x, y, stamp.
        self.history: dict[str, list[tuple[float, float]]] = {t: [] for t in ("/vo/path", "/vo_learned/path")}  # (t, |diff|).
        self.t0: float | None = None
        for topic in SERIES:
            self.create_subscription(PathMsg, topic, lambda msg, topic=topic: self._on_path(topic, msg), 5)
        self.save_client = self.create_client(Trigger, "/vo/save")

    def _on_path(self, topic: str, msg: PathMsg) -> None:
        if not msg.poses:
            return
        data = np.array([(p.pose.position.x, p.pose.position.y, p.header.stamp.sec + p.header.stamp.nanosec * 1e-9) for p in msg.poses])
        with self.lock:
            self.paths[topic] = data
            if self.t0 is None:
                self.t0 = float(data[0, 2])
            reference = self.paths.get("/vo/lidar_path")
            if reference is None or len(reference) < 2 or valid_part(reference)[1] is not None:  # Missing or diverged.
                reference = self.paths.get("/vo/wheel_path")
            if topic in self.history and reference is not None and len(reference):
                diff = float(np.linalg.norm(data[-1, :2] - reference[-1, :2]))
                self.history[topic].append((float(data[-1, 2]) - self.t0, diff))

    def snapshot(self) -> tuple[dict[str, np.ndarray], dict[str, list[tuple[float, float]]]]:
        with self.lock:
            return dict(self.paths), {k: list(v) for k, v in self.history.items()}


def main(args: list[str] | None = None) -> None:
    """Entry point: ROS in a background thread, the window in the main thread."""
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    rclpy.init(args=args)
    node = VOPlotNode()
    spinner = threading.Thread(target=lambda: _spin(node), daemon=True)
    spinner.start()

    fig = plt.figure("MetricVision: odometry comparison", figsize=(8.5, 8.5))
    ax_map = fig.add_axes((0.09, 0.40, 0.88, 0.55))
    ax_err = fig.add_axes((0.09, 0.13, 0.88, 0.19))
    ax_save = fig.add_axes((0.09, 0.02, 0.25, 0.055))
    ax_report = fig.add_axes((0.37, 0.02, 0.30, 0.055))
    status = fig.text(0.70, 0.04, "", fontsize=8, va="center")
    save_button, report_button = Button(ax_save, "Save graph"), Button(ax_report, "Save full report")

    def run_dir() -> Path:
        folder = Path(node.output_dir).resolve() if node.output_dir else Path.cwd() / "results" / "live"
        (folder / "plots").mkdir(parents=True, exist_ok=True)
        return folder

    def save_graph(_: Any) -> None:
        path = run_dir() / "plots" / f"live_{time.strftime('%H%M%S')}.png"
        fig.savefig(path, dpi=150)
        status.set_text(f"saved {path.name}")
        node.get_logger().info(f"saved graph {path}")

    def save_report(_: Any) -> None:
        if not node.save_client.service_is_ready():
            status.set_text("vo_node not running (/vo/save missing)")
            return
        future = node.save_client.call_async(Trigger.Request())
        future.add_done_callback(lambda f: status.set_text((f.result().message if f.result() else "save failed")[-60:]))
        save_graph(None)

    save_button.on_clicked(save_graph)
    report_button.on_clicked(save_report)

    def redraw() -> None:
        if not rclpy.ok():  # Ctrl+C (rclpy's SIGINT handler shut the context down): close the window.
            plt.close(fig)
            return
        draw(ax_map, ax_err, *node.snapshot())
        fig.canvas.draw_idle()

    timer = fig.canvas.new_timer(interval=500)  # Backend-neutral periodic redraw.
    timer.add_callback(redraw)
    timer.start()
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _spin(node: Node) -> None:
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():
            raise


if __name__ == "__main__":
    main()
