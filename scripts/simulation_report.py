"""Comparison report of the two approaches (classical vs learned front end, same rigid-rig back end).

Reads ``<results>/<sequence>/{classical,learned}/`` written by ``metricvision run`` (``scripts/run_simulation.sh``):
``trajectory.tum``, ``reference.tum`` (ground truth: KITTI/EuRoC poses, wheel /odom for the wheelchair bags) and
``summary.json``, and writes::

    <results>/report.html     tables + plots, open in a browser
    <results>/report.md       the same tables as text
    <results>/summary.csv     one row per sequence and approach
    <results>/plots/<sequence>_topdown.png, <sequence>_error.png

Metrics per run: tracking failures, speed, SE(3)-aligned ATE and per-frame RPE (from the run), and, with both paths
anchored at the first frame (no fitting): end-point difference, drift % of the path, VO/reference length ratio
(metric-scale check of the stereo baseline) and final heading difference.

Usage:
    uv run python scripts/simulation_report.py --results results/simulation
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from metricvision.core.geometry import invert_pose, make_pose  # noqa: E402
from metricvision.datasets.rosbag_stereo import interpolate_poses  # noqa: E402

APPROACHES = {
    "classical": ("Classical: FAST/ORB + KLT, rigid-rig LM + window BA", "#1f7ae0"),
    "learned": ("Learned: SuperPoint + LightGlue, rigid-rig LM + window BA", "#8e44ad"),
}


def read_tum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(N,) times and (N, 4, 4) poses."""
    rows = np.loadtxt(path, ndmin=2)
    poses = np.array([make_pose(Rotation.from_quat(r[4:8]).as_matrix(), r[1:4]) for r in rows])
    return rows[:, 0], poses


def path_length(xy: np.ndarray, min_step_m: float = 0.02) -> float:
    """Distance travelled, ignoring jitter below ``min_step_m``."""
    total, anchor = 0.0, xy[0]
    for point in xy[1:]:
        step = float(np.linalg.norm(point - anchor))
        if step >= min_step_m:
            total, anchor = total + step, point
    return total + float(np.linalg.norm(xy[-1] - anchor))


def yaw(poses: np.ndarray) -> np.ndarray:
    return np.asarray(np.degrees(np.unwrap(np.arctan2(poses[:, 1, 0], poses[:, 0, 0]))))


def compare(run: Path) -> dict[str, Any] | None:
    """Metrics of one run folder, plus the anchored series for the plots."""
    if not (run / "trajectory.tum").is_file() or not (run / "reference.tum").is_file():
        return None
    summary = json.loads((run / "summary.json").read_text())
    t, est = read_tum(run / "trajectory.tum")
    rt, ref = read_tum(run / "reference.tum")
    keep = np.concatenate([[True], np.diff(rt) > 0])
    ref_i, inside = interpolate_poses(rt[keep], list(ref[keep]), t)
    t, est = t[inside], est[inside]
    if len(t) < 2:
        return None
    est = np.array([invert_pose(est[0]) @ p for p in est])  # Both anchored at the first common frame.
    ref_i = np.array([invert_pose(ref_i[0]) @ p for p in ref_i])
    diff = np.linalg.norm(est[:, :3, 3] - ref_i[:, :3, 3], axis=1)
    ref_len, est_len = path_length(ref_i[:, :3, 3]), path_length(est[:, :3, 3])
    errors = summary.get("errors") or {}
    return {
        "frames": summary["num_frames"],
        "failures_pct": 100.0 * summary["num_tracking_failures"] / max(summary["num_frames"], 1),
        "fps": summary["fps"],
        "ate_m": errors.get("ate_rmse_m", float("nan")),
        "rpe_m": errors.get("rpe_trans_rmse_m", float("nan")),
        "ref_length_m": ref_len,
        "length_ratio": est_len / ref_len if ref_len > 0.05 else float("nan"),
        "end_diff_m": float(diff[-1]),
        "drift_pct": 100.0 * float(diff[-1]) / ref_len if ref_len > 0.05 else float("nan"),
        "heading_diff_deg": float(abs((yaw(est)[-1] - yaw(ref_i)[-1] + 180.0) % 360.0 - 180.0)),
        "_t": t - t[0],
        "_est": est,
        "_ref": ref_i,
        "_diff": diff,
    }


def plane_axes(ref: np.ndarray) -> tuple[int, int]:
    """The two axes of the ground plane: x/y for the wheelchair (z up), x/z for KITTI/EuRoC camera frames."""
    spread = np.ptp(ref[:, :3, 3], axis=0)
    return (0, 1) if spread[2] <= min(spread[0], spread[1]) else (0, 2)


def plot(sequence: str, runs: dict[str, dict[str, Any]], out: Path) -> tuple[str, str]:
    first = next(iter(runs.values()))
    a, b = plane_axes(first["_ref"])
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(first["_ref"][:, a, 3], first["_ref"][:, b, 3], "k-", lw=2.4, label=f"reference ({first['ref_length_m']:.1f} m)")
    for name, m in runs.items():
        label, colour = APPROACHES[name]
        ax.plot(m["_est"][:, a, 3], m["_est"][:, b, 3], color=colour, lw=1.5, label=f"{label} (ATE {m['ate_m']:.3f} m)")
    ax.plot(0, 0, "ko")
    ax.set(title=f"{sequence}: top-down, anchored at the first frame", xlabel="xyz"[a] + " [m]", ylabel="xyz"[b] + " [m]")
    ax.axis("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    topdown = out / f"{sequence}_topdown.png"
    fig.savefig(topdown, dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    for name, m in runs.items():
        label, colour = APPROACHES[name]
        ax.plot(m["_t"], m["_diff"], color=colour, label=label)
    ax.set(title=f"{sequence}: position difference to the reference (anchored, no fitting)", xlabel="time [s]", ylabel="[m]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    error = out / f"{sequence}_error.png"
    fig.savefig(error, dpi=130)
    plt.close(fig)
    return topdown.name, error.name


COLUMNS = [
    ("frames", "Frames", "{:.0f}"),
    ("failures_pct", "Tracking lost", "{:.1f} %"),
    ("fps", "Speed", "{:.1f} fps"),
    ("ate_m", "ATE (SE3)", "{:.3f} m"),
    ("rpe_m", "RPE / frame", "{:.4f} m"),
    ("end_diff_m", "End difference", "{:.3f} m"),
    ("drift_pct", "Drift", "{:.2f} %"),
    ("length_ratio", "Length ratio", "{:.3f}"),
    ("heading_diff_deg", "Heading diff", "{:.1f}°"),
]


def fmt(value: float, pattern: str) -> str:
    return "—" if value is None or (isinstance(value, float) and not np.isfinite(value)) else pattern.format(value)


def summary_chart(rows: list[dict[str, Any]], out: Path) -> str | None:
    """One figure for all sequences: ATE and drift per sequence, classical next to learned (the headline proof)."""
    sequences = sorted({r["sequence"] for r in rows})
    if not sequences:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    width = 0.38
    x = np.arange(len(sequences))
    for k, (key, title) in enumerate((("ate_m", "ATE (SE3-aligned) [m]"), ("drift_pct", "End drift [% of path]"), ("fps", "Speed [frames/s]"))):
        ax = axes[k]
        for j, (name, (_label, colour)) in enumerate(APPROACHES.items()):
            values = [next((r[key] for r in rows if r["sequence"] == seq and r["approach"] == name), np.nan) for seq in sequences]
            bars = ax.bar(x + (j - 0.5) * width, values, width, color=colour, label=name)
            ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
        ax.set_xticks(x, [s.replace("wheelchair_", "wc_") for s in sequences], rotation=20, fontsize=8)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        if key == "ate_m":
            ax.set_yscale("log")
    axes[0].legend(fontsize=8)
    fig.suptitle("Classical vs learned front end, same rigid-rig back end (lower ATE / drift is better)")
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=130)
    plt.close(fig)
    return "overview.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results/simulation"))
    args = parser.parse_args()
    plots = args.results / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    md = ["# Simulation results: classical vs learned front end (same rigid-rig back end)", ""]
    md += [f"- **{name}**: {label}" for name, (label, _) in APPROACHES.items()]
    md += ["- Reference: KITTI/EuRoC ground truth; wheel odometry `/odom` for the wheelchair bags (it slips and drifts too).", ""]
    sections, rows = [], []
    for sequence_dir in sorted(d for d in args.results.iterdir() if d.is_dir() and d.name not in ("plots", "logs")):
        runs = {name: m for name in APPROACHES if (m := compare(sequence_dir / name)) is not None}
        if not runs:
            continue
        topdown, error = plot(sequence_dir.name, runs, plots)
        header = "| Approach | " + " | ".join(title for _, title, _ in COLUMNS) + " |"
        md += [
            f"## {sequence_dir.name} (reference path {next(iter(runs.values()))['ref_length_m']:.1f} m)",
            "",
            header,
            "|" + "---|" * (len(COLUMNS) + 1),
        ]
        table_rows = []
        for name, m in runs.items():
            cells = [fmt(m[key], pattern) for key, _, pattern in COLUMNS]
            md.append(f"| {name} | " + " | ".join(cells) + " |")
            table_rows.append(f"<tr><td>{html.escape(APPROACHES[name][0])}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
            rows.append({"sequence": sequence_dir.name, "approach": name, **{k: m[k] for k, _, _ in COLUMNS}})
        md += ["", f"![top-down](plots/{topdown})", f"![error](plots/{error})", ""]
        sections.append(
            f"<h2>{html.escape(sequence_dir.name)}</h2><table><tr><th>Approach</th>"
            + "".join(f"<th>{t}</th>" for _, t, _ in COLUMNS)
            + "</tr>"
            + "".join(table_rows)
            + f'</table><div class="plots"><img src="plots/{topdown}"><img src="plots/{error}"></div>'
        )
    overview = summary_chart(rows, plots)
    if overview:
        header_end = md.index("", 2) + 1  # After the title and the approach list.
        md[header_end:header_end] = [f"![overview](plots/{overview})", ""]
        sections.insert(0, f'<h2>Overview</h2><div class="plots" style="grid-template-columns:1fr"><img src="plots/{overview}"></div>')
    (args.results / "report.md").write_text("\n".join(md) + "\n")
    with (args.results / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence", "approach"] + [k for k, _, _ in COLUMNS])
        writer.writeheader()
        writer.writerows(rows)
    legend = "".join(f"<li><b>{n}</b>: {html.escape(label)}</li>" for n, (label, _) in APPROACHES.items())
    (args.results / "report.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Simulation results</title><style>"
        "body{font:14px/1.45 system-ui,sans-serif;max-width:1200px;margin:0 auto;padding:20px 16px;color:#1d1d21;background:#f7f7f8}"
        "table{border-collapse:collapse;width:100%;background:#fff;margin:8px 0}th,td{border-bottom:1px solid #e3e3e8;padding:6px 8px;"
        "text-align:left;font-variant-numeric:tabular-nums}th{color:#6b6b76;font-weight:500;font-size:12px}"
        ".plots{display:grid;grid-template-columns:1fr 1.3fr;gap:10px}.plots img{width:100%;background:#fff;border:1px solid #e3e3e8}"
        "@media (max-width:800px){.plots{grid-template-columns:1fr}}"
        "</style></head><body><h1>Simulation results: classical vs learned front end</h1><p>Same rigid-rig back end (LM + "
        f"sliding-window BA). Reference: KITTI/EuRoC ground truth, wheel odometry for the wheelchair bags.</p><ul>{legend}</ul>"
        + "".join(sections)
        + "</body></html>"
    )
    print(f"wrote {args.results}/report.html, report.md, summary.csv, plots/")


if __name__ == "__main__":
    main()
