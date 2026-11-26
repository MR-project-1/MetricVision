"""Interactive web report of one live run (``report.html``) and an index of all runs (``index.html``).

Self-contained HTML: the data is embedded as JSON and drawn by a small inline SVG chart script, so the page
works offline (no CDN, no plotting package). Hover any chart for values; the top-down map keeps equal axes.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from metricvision.core.geometry import invert_pose
from metricvision.core.types import Trajectory


def _series(trajectory: Trajectory, t0: float, anchor: bool) -> dict[str, list[float]]:
    """Time, x, y, z, heading (deg, unwrapped) and planar speed of a trajectory."""
    if len(trajectory) == 0:
        return {"t": [], "x": [], "y": [], "z": [], "yaw": [], "speed": []}
    inv = invert_pose(trajectory.poses[0]) if anchor else np.eye(4)
    poses = np.array([inv @ p for p in trajectory.poses])
    t = np.asarray(trajectory.timestamps) - t0
    yaw = np.degrees(np.unwrap(np.arctan2(poses[:, 1, 0], poses[:, 0, 0])))
    speed = np.zeros(len(t))
    if len(t) > 2:  # Central difference over about half a second, robust to frame-rate jitter.
        k = max(1, int(round(0.25 / max(np.median(np.diff(t)), 1e-3))))
        lo, hi = np.clip(np.arange(len(t)) - k, 0, len(t) - 1), np.clip(np.arange(len(t)) + k, 0, len(t) - 1)
        dt = np.maximum(t[hi] - t[lo], 1e-6)
        speed = np.linalg.norm(poses[hi, :2, 3] - poses[lo, :2, 3], axis=1) / dt

    def r(a: np.ndarray, n: int = 4) -> list[float]:
        return [round(float(v), n) for v in a]

    return {
        "t": r(t, 3),
        "x": r(poses[:, 0, 3]),
        "y": r(poses[:, 1, 3]),
        "z": r(poses[:, 2, 3]),
        "yaw": r(yaw, 2),
        "speed": r(speed, 3),
    }


def _fmt(value: Any, unit: str = "", digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}{unit}" if isinstance(value, float) else f"{value}{unit}"


_REF_COLOURS = {"lidar": "--li", "wheel": "--od"}


def _rate(trajectory: Trajectory) -> float:
    ts = trajectory.timestamps
    return (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else 0.0


def write_report_html(
    folder: Path,
    vo: Trajectory,
    refs: list[tuple[str, str, Trajectory, Trajectory]],
    metrics: dict[str, Any],
    extra: list[tuple[str, Trajectory]] | None = None,
    main_name: str = "classical",
) -> Path:
    """Write ``folder/report.html``.

    Args:
        folder: Run folder.
        vo: VO trajectory.
        refs: (key, label, raw trajectory, trajectory at the VO stamps anchored at the first VO frame), best first;
            the first one is the reference of the difference plots and the headline numbers.
        metrics: From :func:`metricvision.evaluation.live_report.write_live_report`.
    """
    t0 = vo.timestamps[0]
    vo_s = _series(vo, t0, anchor=True)
    odom_at_vo = refs[0][3] if refs else Trajectory()
    od_s = _series(odom_at_vo, t0, anchor=False)  # Already anchored at the first VO frame.
    ref_data = [{"n": label, "c": _REF_COLOURS.get(key, "--od"), "s": _series(at_vo, t0, anchor=False)} for key, label, _, at_vo in refs]
    diff: dict[str, list[float]] = {"t": [], "pos": [], "yaw": []}
    if len(odom_at_vo):
        index = {round(t, 6): i for i, t in enumerate(vo.timestamps)}
        for j, t in enumerate(odom_at_vo.timestamps):
            i = index.get(round(t, 6))
            if i is None:
                continue
            diff["t"].append(round(t - t0, 3))
            diff["pos"].append(round(float(np.hypot(vo_s["x"][i] - od_s["x"][j], vo_s["y"][i] - od_s["y"][j])), 4))
            d = (vo_s["yaw"][i] - od_s["yaw"][j] + 180.0) % 360.0 - 180.0
            diff["yaw"].append(round(float(d), 2))
    by_ref = metrics.get("by_reference", {})
    cards = [
        ("Reference", str(metrics.get("reference", "none"))),
        ("Duration", _fmt(metrics.get("duration_s"), " s", 1)),
        ("Stereo VO path", _fmt(metrics.get("vo_length_m"), " m", 2)),
        *[(f"{label} path", _fmt(by_ref.get(key, {}).get("odom_length_m"), " m", 2)) for key, label, _, _ in refs],
        ("Length ratio VO / reference", _fmt(metrics.get("length_ratio"), "", 3)),
        ("End position difference", _fmt(metrics.get("end_error_m"), " m", 3)),
        ("Drift (end diff / path)", _fmt(metrics.get("drift_pct"), " %", 1)),
        ("Final heading difference", _fmt(metrics.get("yaw_error_deg"), "°", 1)),
        ("ATE (SE3-aligned RMSE)", _fmt(metrics.get("ate_se3_m"), " m", 3)),
        ("Max position difference", _fmt(metrics.get("max_error_m"), " m", 3)),
        ("VO poses / rate", f"{len(vo)} / {_rate(vo):.1f} Hz"),
        *[(f"{label} msgs / rate", f"{len(raw)} / {_rate(raw):.1f} Hz") for _, label, raw, _ in refs],
        ("Frames compared", _fmt(metrics.get("frames"))),
    ]
    frames: dict[str, list[float]] = {}
    frames_file = (folder / "data" / "frames.csv") if (folder / "data" / "frames.csv").is_file() else (folder / "frames.csv")
    if frames_file.is_file():
        rows = np.atleast_2d(np.loadtxt(frames_file, delimiter=",", skiprows=1, ndmin=2))
        if rows.size:
            frames = {
                "t": [round(float(v - t0), 3) for v in rows[:, 0]],
                "ok": [int(v) for v in rows[:, 1]],
                "tracked": [int(v) for v in rows[:, 2]],
                "depth": [int(v) for v in rows[:, 3]],
                "inliers": [int(v) for v in rows[:, 4]],
                "new": [int(v) for v in rows[:, 5]],
                "bl": [round(float(v), 1) for v in rows[:, 6]],
                "br": [round(float(v), 1) for v in rows[:, 7]],
            }
    if "tracking_ok_pct" in metrics:
        cards.insert(0, ("Tracking OK", f"{metrics['tracking_ok_pct']:.0f} % of frames"))
    vos = [{"n": f"Stereo VO {main_name}", "c": "--vo", "s": vo_s}] + [
        {"n": f"Stereo VO {name}", "c": "--lv", "s": _series(trajectory, t0, anchor=True)} for name, trajectory in (extra or []) if len(trajectory)
    ]
    data = {"vo": vo_s, "vos": vos, "refs": ref_data, "diff": diff, "frames": frames}
    legend = "".join(f'<span><i class="sw" style="background:var({r["c"]})"></i>{html.escape(str(r["n"]))}</span>' for r in [*vos, *ref_data])
    approaches = metrics.get("approaches") or {}
    approach_table = ""
    if len(approaches) > 1:
        cols = [
            ("tracking_ok_pct", "Tracking OK", " %", 0),
            ("vo_length_m", "Path", " m", 2),
            ("length_ratio", "Length ratio", "", 3),
            ("end_error_m", "End diff", " m", 3),
            ("drift_pct", "Drift", " %", 1),
            ("yaw_error_deg", "Heading diff", "°", 1),
            ("ate_se3_m", "ATE", " m", 3),
        ]
        head = "".join(f"<th>{t}</th>" for _, t, _, _ in cols)
        body = "".join(
            f"<tr><td>{html.escape(name)}</td>" + "".join(f"<td>{_fmt(m.get(k), u, d)}</td>" for k, _, u, d in cols) + "</tr>"
            for name, m in approaches.items()
        )
        approach_table = (
            f'<div class="panel wide" style="margin-bottom:14px"><h2>Approaches on the same frames vs {html.escape(str(metrics.get("reference", "reference")))}</h2>'
            f'<table class="cmp"><tr><th>Approach</th>{head}</tr>{body}</table></div>'
        )
    video = (
        '<div class="panel wide" style="margin-bottom:14px"><h2>Run video: what the VO saw (keypoints, stereo matches, map, numbers)</h2>'
        '<video src="video.webm" controls muted playsinline style="width:100%;border-radius:8px;background:#000"></video></div>'
        if (folder / "video.webm").is_file()
        else ""
    )
    links = [
        ("summary.md", "summary.md"),
        ("run_info.yaml", "run_info.yaml"),
        ("video.webm", "video.webm"),
        ("plots/trajectory_topdown.png", "plot: top-down"),
        ("plots/position_heading.png", "plot: position & heading"),
        ("plots/difference.png", "plot: difference"),
        ("plots/tracking.png", "plot: tracking"),
        ("data/", "data/ (TUM, frames.csv, metrics.json)"),
        ("calibration/", "calibration/"),
        ("bag/", "bag/"),
    ]
    files = " · ".join(f'<a href="{href}">{html.escape(text)}</a>' for href, text in links if (folder / href.rstrip("/")).exists())
    notes = "".join(f'<div class="note">{html.escape(n)}</div>' for n in metrics.get("notes", []))
    if "ate_note" in metrics:
        notes += f'<div class="note">{html.escape(metrics["ate_note"])}</div>'
    card_html = "".join(f'<div class="card"><div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div></div>' for k, v in cards)
    started = datetime.fromtimestamp(t0).strftime("%Y-%m-%d %H:%M:%S")
    page = (
        _PAGE.replace("__TITLE__", html.escape(folder.name))
        .replace("__STARTED__", started)
        .replace("__NOTES__", notes)
        .replace("__CARDS__", card_html)
        .replace("__LEGEND__", legend)
        .replace("__VIDEO__", approach_table + video)
        .replace("__FILES__", files)
        .replace("__REF__", html.escape(str(metrics.get("reference", "reference"))))
        .replace("__DATA__", json.dumps(data, separators=(",", ":")))
    )
    out = folder / "report.html"
    out.write_text(page)
    return out


def write_index(live_root: Path) -> Path:
    """Write ``live_root/index.html``: one row per run folder that has a ``metrics.json``."""
    rows = []

    def metrics_file(d: Path) -> Path:
        return d / "data" / "metrics.json" if (d / "data" / "metrics.json").is_file() else d / "metrics.json"

    for folder in sorted((d for d in live_root.iterdir() if d.is_dir() and metrics_file(d).is_file()), reverse=True):
        m = json.loads(metrics_file(folder).read_text())
        link = f'<a href="{html.escape(folder.name)}/report.html">{html.escape(folder.name)}</a>'
        cells = [
            link,
            _fmt(m.get("duration_s"), " s", 1),
            _fmt(m.get("vo_length_m"), " m", 2),
            _fmt(m.get("odom_length_m"), " m", 2),
            _fmt(m.get("length_ratio"), "", 3),
            _fmt(m.get("end_error_m"), " m", 3),
            _fmt(m.get("drift_pct"), " %", 1),
            _fmt(m.get("yaw_error_deg"), "°", 1),
            "⚠" if m.get("notes") else "",
        ]
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    head = ["Run", "Duration", "VO path", "Odom path", "Length ratio", "End diff", "Drift", "Heading diff", "Notes"]
    table = "<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    out = live_root / "index.html"
    out.write_text(_INDEX.replace("__TABLE__", table))
    return out


_STYLE = """
:root{--bg:#f7f7f8;--panel:#fff;--ink:#1d1d21;--muted:#6b6b76;--line:#e3e3e8;--vo:#1f7ae0;--od:#1a9b55;--li:#e07b00;--lv:#8e44ad;--err:#d6453d;--warn-bg:#fff4e0;--warn:#8a5300}
@media (prefers-color-scheme:dark){:root{--bg:#17171a;--panel:#212126;--ink:#ececf1;--muted:#9a9aa6;--line:#34343c;--vo:#4ea1ff;--od:#3fd17f;--li:#ffa23a;--lv:#c39bd3;--err:#ff6b61;--warn-bg:#3a2c12;--warn:#ffc46b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1200px;margin:0 auto;padding:20px 16px 48px}h1{font-size:22px;margin:0 0 2px}.sub{color:var(--muted);margin-bottom:16px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px;margin:14px 0 18px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px}.card .k{color:var(--muted);font-size:12px}.card .v{font-size:19px;font-weight:600;font-variant-numeric:tabular-nums}
.note{background:var(--warn-bg);color:var(--warn);border-radius:8px;padding:9px 12px;margin:8px 0}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media (max-width:820px){.grid{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px}.panel h2{font-size:14px;margin:0 0 6px}.wide{grid-column:1/-1}
.legend{display:flex;gap:14px;color:var(--muted);font-size:12px;margin-bottom:4px}.sw{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:5px;vertical-align:middle}
svg{width:100%;display:block;overflow:visible}svg text{fill:var(--muted);font-size:11px}.ax{stroke:var(--line)}
.tip{position:fixed;pointer-events:none;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:5px 8px;font-size:12px;box-shadow:0 2px 8px #0003;display:none;font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;font-variant-numeric:tabular-nums}th{color:var(--muted);font-weight:500;font-size:12px}a{color:var(--vo)}
.explain{color:var(--muted);font-size:12px;margin-top:18px}.files{color:var(--muted);font-size:12px;margin:0 0 14px}.cmp td,.cmp th{padding:6px 8px}
"""

_PAGE = (
    """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VO run __TITLE__</title><style>"""
    + _STYLE
    + """</style></head><body><main>
<h1>Stereo VO vs reference odometry</h1><div class="sub">Run __TITLE__ · started __STARTED__ · <a href="../index.html">all runs</a></div>
__NOTES__<div class="cards">__CARDS__</div>
__VIDEO__
<div class="files">Files in this run: __FILES__</div>
<div class="grid">
<div class="panel wide"><h2>Top-down path (all start at the first VO frame, equal axes)</h2><div class="legend">__LEGEND__</div><div id="map"></div></div>
<div class="panel"><h2>x over time [m]</h2><div id="x"></div></div>
<div class="panel"><h2>y over time [m]</h2><div id="y"></div></div>
<div class="panel"><h2>Heading over time [°]</h2><div id="yaw"></div></div>
<div class="panel"><h2>Speed [m/s]</h2><div id="speed"></div></div>
<div class="panel"><h2>Position difference VO − __REF__ [m]</h2><div id="dpos"></div></div>
<div class="panel"><h2>Heading difference VO − __REF__ [°]</h2><div id="dyaw"></div></div>
<div class="panel wide" id="trk"><h2>Tracking health per frame (red band = tracking lost, VO coasting)</h2><div class="legend"><span><i class="sw" style="background:var(--vo)"></i>PnP inliers</span><span><i class="sw" style="background:var(--od)"></i>points with stereo depth</span><span><i class="sw" style="background:var(--muted)"></i>new detections</span></div><div id="inl"></div></div>
<div class="panel wide" id="brp"><h2>Image brightness (mean grey value, 0–255): a dark or covered camera shows here</h2><div class="legend"><span><i class="sw" style="background:var(--vo)"></i>left</span><span><i class="sw" style="background:var(--err)"></i>right</span></div><div id="bright"></div></div>
<div class="panel wide"><h2>Height of the VO base frame z [m] (should stay near 0 on a flat floor)</h2><div id="z"></div></div>
</div>
<p class="explain">Reference = lidar-inertial odometry (DLIO, /Odometry) when it was recorded, else wheel odometry. Neither is perfect ground truth: wheels slip and record nothing when the chair is pushed by hand; DLIO drifts slowly. The numbers measure agreement. Length ratio checks the stereo baseline (metric scale): about 1.0 means the scale is right. Drift = end position difference / odometry path length.</p>
</main><div class="tip" id="tip"></div>
<script>
const D=__DATA__;
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const tip=document.getElementById('tip');
function ext(arrs){let lo=Infinity,hi=-Infinity;for(const a of arrs)for(const v of a){if(v<lo)lo=v;if(v>hi)hi=v}if(!isFinite(lo)){lo=0;hi=1}if(hi-lo<1e-9){lo-=0.5;hi+=0.5}return[lo,hi]}
function ticks(lo,hi,n){const s=Math.pow(10,Math.floor(Math.log10((hi-lo)/n)));const m=[1,2,5,10].find(m=>(hi-lo)/(s*m)<=n)*s;const out=[];for(let v=Math.ceil(lo/m)*m;v<=hi+1e-9;v+=m)out.push(+v.toFixed(10));return out}
function chart(id,series,o={}){
  const el=document.getElementById(id);const W=el.clientWidth||560,H=o.h||220,L=48,R=10,T=8,B=26;
  let [x0,x1]=ext(series.map(s=>s.x)),[y0,y1]=ext(series.map(s=>s.y));
  if(o.zeroY){y0=Math.min(y0,0);y1=Math.max(y1,0)}
  const py=(y1-y0)*0.06;y0-=py;y1+=py;
  if(o.equal){const sx=(x1-x0)/(W-L-R),sy=(y1-y0)/(H-T-B),s=Math.max(sx,sy);const cx=(x0+x1)/2,cy=(y0+y1)/2;x0=cx-s*(W-L-R)/2;x1=cx+s*(W-L-R)/2;y0=cy-s*(H-T-B)/2;y1=cy+s*(H-T-B)/2}
  const X=v=>L+(v-x0)/(x1-x0)*(W-L-R),Y=v=>H-B-(v-y0)/(y1-y0)*(H-T-B);
  let g='';for(const v of ticks(y0,y1,5))g+=`<line class="ax" x1="${L}" x2="${W-R}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${L-6}" y="${Y(v)+4}" text-anchor="end">${v}</text>`;
  for(const v of ticks(x0,x1,o.equal?8:7))g+=`<line class="ax" y1="${T}" y2="${H-B}" x1="${X(v)}" x2="${X(v)}"/><text x="${X(v)}" y="${H-8}" text-anchor="middle">${v}</text>`;
  let p='';if(o.bands){const xs=series[0].x;let i=0;while(i<xs.length){if(o.bands[i]){i++;continue}let j=i;while(j<xs.length&&!o.bands[j])j++;const xa=X(xs[i]),xb=X(xs[Math.min(j,xs.length-1)]);p+=`<rect x="${xa}" y="${T}" width="${Math.max(1,xb-xa)}" height="${H-T-B}" fill="${css('--err')}" opacity="0.15"/>`;i=j}}
  for(const s of series){if(!s.x.length)continue;let d='';for(let i=0;i<s.x.length;i++)d+=(i?'L':'M')+X(s.x[i]).toFixed(1)+','+Y(s.y[i]).toFixed(1);p+=`<path d="${d}" fill="none" stroke="${css(s.c)}" stroke-width="${s.w||1.8}" stroke-linejoin="round"/>`;if(o.equal)p+=`<circle cx="${X(s.x[0])}" cy="${Y(s.y[0])}" r="4" fill="${css(s.c)}"/><circle cx="${X(s.x.at(-1))}" cy="${Y(s.y.at(-1))}" r="4" fill="none" stroke="${css(s.c)}" stroke-width="2"/>`}
  el.innerHTML=`<svg viewBox="0 0 ${W} ${H}" height="${H}">${g}${p}<g id="${id}h"></g><rect x="${L}" y="${T}" width="${W-L-R}" height="${H-T-B}" fill="transparent"/></svg>`;
  const svg=el.querySelector('svg'),hg=svg.querySelector('g:last-of-type');
  svg.addEventListener('mousemove',e=>{const r=svg.getBoundingClientRect(),mx=(e.clientX-r.left)*W/r.width,my=(e.clientY-r.top)*H/r.height;let lines=[],marks='';
    for(const s of series){if(!s.x.length)continue;let bi=0,bd=Infinity;for(let i=0;i<s.x.length;i++){const d=o.equal?Math.hypot(X(s.x[i])-mx,Y(s.y[i])-my):Math.abs(X(s.x[i])-mx);if(d<bd){bd=d;bi=i}}
      marks+=`<circle cx="${X(s.x[bi])}" cy="${Y(s.y[bi])}" r="4" fill="${css(s.c)}"/>`;lines.push(`<span style="color:${css(s.c)}">●</span> ${s.n}: `+(o.equal?`t=${s.t[bi]} s, x=${s.x[bi]}, y=${s.y[bi]}`:`t=${s.x[bi]} s → ${s.y[bi]}`))}
    hg.innerHTML=marks;tip.innerHTML=lines.join('<br>');tip.style.display='block';tip.style.left=Math.min(e.clientX+14,innerWidth-260)+'px';tip.style.top=(e.clientY+14)+'px'});
  svg.addEventListener('mouseleave',()=>{hg.innerHTML='';tip.style.display='none'});
}
function draw(){const v=D.vo,f=D.diff,R=D.refs;
  const VS=D.vos;
  chart('map',R.map((r,i)=>({n:r.n,x:r.s.x,y:r.s.y,t:r.s.t,c:r.c,w:i?2:3})).concat(VS.map(o=>({n:o.n,x:o.s.x,y:o.s.y,t:o.s.t,c:o.c}))),{equal:true,h:420});
  const ts=(k)=>R.map((r,i)=>({n:r.n,x:r.s.t,y:r.s[k],c:r.c,w:i?1.8:2.6})).concat(VS.map(o=>({n:o.n,x:o.s.t,y:o.s[k],c:o.c})));
  chart('x',ts('x'));chart('y',ts('y'));chart('yaw',ts('yaw'));chart('speed',ts('speed'),{zeroY:true});
  chart('dpos',[{n:'|xy difference| [m]',x:f.t,y:f.pos,c:'--err'}],{zeroY:true});chart('dyaw',[{n:'heading difference [°]',x:f.t,y:f.yaw,c:'--err'}],{zeroY:true});
  chart('z',[{n:'VO z',x:v.t,y:v.z,c:'--vo'}],{zeroY:true,h:160});
  const F=D.frames;if(!F.t){document.getElementById('trk').style.display='none';document.getElementById('brp').style.display='none';return}
  chart('inl',[{n:'new detections',x:F.t,y:F.new,c:'--muted',w:1.2},{n:'with stereo depth',x:F.t,y:F.depth,c:'--od',w:1.4},{n:'PnP inliers',x:F.t,y:F.inliers,c:'--vo'}],{zeroY:true,h:240,bands:F.ok});
  chart('bright',[{n:'left',x:F.t,y:F.bl,c:'--vo'},{n:'right',x:F.t,y:F.br,c:'--err'}],{zeroY:true,h:160});}
draw();let rt;addEventListener('resize',()=>{clearTimeout(rt);rt=setTimeout(draw,150)});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',draw);
</script></body></html>"""
)

_INDEX = (
    """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Live VO runs</title><style>"""
    + _STYLE
    + """</style></head><body><main><h1>Live VO runs</h1><div class="sub">Stereo VO vs wheel odometry (/odom), newest first. Open a run for the plots.</div>
<div style="overflow-x:auto">__TABLE__</div></main></body></html>"""
)
