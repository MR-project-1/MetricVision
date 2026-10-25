# MetricVision
Feature extractors for visual odometry, measured end to end

Benchmark visual odometry front ends (feature extraction + matching) inside one
fixed stereo VO back end, so trajectory differences come from the extractor and
nothing else.
 
Status (mid submission, Oct 2026): two stereo VO pipelines sharing one evaluation path:

- **Sparse pipeline** (`pipeline: sparse`): FAST + ORB stereo matching along epipolar lines,
  triangulation with first-order depth covariance, KLT tracking with a circular L/R check
  (3D-2D correspondences per frame), PnP + RANSAC initial pose, and a covariance-weighted
  **rigid-rig Levenberg-Marquardt** refinement with analytic Jacobians (left + right residuals).
- **Dense pipeline** (`pipeline: dense`): pairwise ORB/SIFT matching against keyframes,
  fixed SGBM stereo depth, PnP + RANSAC, with the same LM available as refinement.

Evaluated with evo (ATE/RPE) and a port of the KITTI devkit (t_rel, r_rel) on KITTI 00,
EuRoC MH_01 and a built-in synthetic sequence for smoke testing. Results and the mid report are in
[docs/mid_report.md](docs/mid_report.md) ([LaTeX](docs/mid_report.tex)).
SuperPoint+LightGlue, ALIKED+LightGlue, MASt3R, sliding-window bundle adjustment and the ROS 2 node come next.
 
## Project layout
 
```
metricvision/
├── pyproject.toml            # Dependencies and tool settings (uv, ruff, mypy, pytest)
├── uv.lock                   # Exact dependency versions, shared by laptop and Jetson
├── .python-version           # Python 3.10 (ROS 2 Humble and JetPack 6 both use it)
├── .devcontainer/
│   ├── laptop/               # x86_64 + CUDA image
│   └── jetson/               # Jetson AGX Orin (L4T / JetPack 6) image
├── configs/                  # One YAML per dataset
├── src/metricvision/
│   ├── core/                 # Data types, SE(3) helpers (exp map), stereo triangulation + covariance, registry
│   ├── datasets/             # StereoDataset ABC + EuRoC, KITTI, synthetic loaders
│   ├── frontend/             # FeatureMatcher ABC + ORB, SIFT; StereoFrontend ABC + FAST/ORB/KLT  <- variable under test
│   ├── depth/                # DepthEstimator ABC + SGBM (dense pipeline only)
│   ├── backend/              # PnP + RANSAC, rigid-rig LM (lm.py), keyframe VO, frame-to-frame sparse VO
│   ├── evaluation/           # Front-end metrics, ATE/RPE via evo, KITTI devkit metric, output writers
│   ├── config.py             # YAML -> typed config
│   ├── pipeline.py           # Runs one (sequence, front end) experiment
│   └── cli.py                # `metricvision run` / `metricvision list`
├── scripts/                  # run_experiments.sh (full matrix), make_report_assets.py (tables + figures)
├── docs/                     # Mid report (Markdown + LaTeX), generated tables and figures
└── tests/                    # Unit + end-to-end tests (no downloads needed)
```
 
## Setup
 
### Option A: uv directly on the host
 
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # Install uv once
uv sync --all-groups                              # Create .venv from uv.lock
uv run pytest                                     # Verify: all tests should pass
```
 
### Option B: laptop dev container
 
Prerequisites: Docker, the NVIDIA Container Toolkit, an NVIDIA driver ≥ 550,
and VS Code with the Dev Containers extension.
 
1. Put datasets under `~/datasets` on the host. It is mounted at `/data`.
2. Run **Dev Containers: Reopen in Container** and choose **metricvision (laptop, CUDA)**.
3. The container runs `uv sync --all-groups` on creation. Check with
   `metricvision run configs/synthetic.yaml`.
   
### Option C: Jetson dev container
 
1. On the Jetson, run `cat /etc/nv_tegra_release`. Set `L4T_TAG` in
   `.devcontainer/jetson/devcontainer.json` to match (e.g. `R36 … REVISION: 4.0` → `r36.4.0`).
2. From the laptop, use **Remote-SSH** to connect to the Jetson and open the cloned repo.
3. Run **Reopen in Container** and choose **metricvision (Jetson AGX)**.
4. Run the synthetic smoke test. It exercises the entire pipeline without any data.
Both images use Ubuntu 22.04 with its system Python 3.10 and the same
`uv.lock`, so the environment is identical on both machines. Adding ROS 2
Humble's `rclpy` later works because it is built for that same Python.
 
## Datasets
 
```
~/datasets/euroc/MH_01_easy/mav0/...     # ASL format zip from the EuRoC page, extracted
~/datasets/kitti/sequences/00/...        # KITTI odometry grayscale + calibration
~/datasets/kitti/poses/00.txt            # KITTI ground truth (sequences 00-10)
```
 
EuRoC images are distorted, so the loader rectifies them with the calibration
in each `sensor.yaml`. EuRoC ground truth describes the IMU body frame. The
pipeline therefore converts camera poses to the body frame before evaluating
and writing trajectories. KITTI images are already rectified, and its ground
truth is in the left camera frame.
 
## Running experiments
 
```bash
metricvision list                                              # Show available components
metricvision run configs/synthetic.yaml                        # Smoke test, no download
metricvision run configs/euroc.yaml --dataset-root /data/euroc/MH_01_easy
metricvision run configs/euroc.yaml --dataset-root /data/euroc/MH_01_easy --matcher sift
metricvision run configs/kitti.yaml --dataset-root /data/kitti --max-frames 500
metricvision run configs/kitti.yaml --dataset-root /data/kitti --pipeline sparse --refine lm --run-name sparse_lm
```

`--pipeline {dense,sparse}` picks the VO pipeline, `--refine {none,opencv,lm}` the pose
refinement after PnP + RANSAC, and `--run-name` the output folder (use it for ablations so
runs don't overwrite each other). The full mid-submission matrix (6 runs per sequence) and
the report tables/figures are produced by:

```bash
KITTI_ROOT=/data/kitti EUROC_ROOT=/data/euroc/MH_01_easy JOBS=3 scripts/run_experiments.sh
```
 
Outside a container, prefix commands with `uv run`. `--matcher` keeps the
shared parameters (`max_keypoints`, `ratio`, `mutual_check`), so the keypoint
budget stays equal when you switch extractors.
 
Each run writes to `results/<sequence>/<run name>/` (the run name defaults to the front-end name):
 
| File | Contents |
|---|---|
| `summary.json` | ATE/RPE, tracking failures, mean matches, inlier ratio, runtimes, FPS |
| `frame_stats.csv` | Per-frame matches, inliers, flags and timings (for plots) |
| `trajectory.tum` | Body-frame trajectory, TUM format |
| `trajectory.kitti` | Same trajectory, KITTI format |
 
To plot with evo (use `--save_plot out.pdf` inside containers without a display):
 
```bash
evo_ape euroc /data/euroc/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv \
    results/euroc_MH_01_easy/orb/trajectory.tum --align --plot --plot_mode xy
evo_ape kitti /data/kitti/poses/00.txt results/kitti_00/orb/trajectory.kitti --align --plot --plot_mode xz
```
 
## Reproducing the mid-submission results

Every number and figure in [docs/mid_report.pdf](docs/mid_report.pdf) comes from the commands
below (run from the repository root). On the host, prefix `uv run ...` with `env -u PYTHONPATH`
if ROS 2 is sourced in your shell.

**1. Data (single sequences only, via HTTP range requests):**

```bash
uv run --with remotezip python scripts/download_datasets.py kitti --sequence 00 --out ~/datasets/kitti
uv run --with remotezip python scripts/download_datasets.py euroc --sequence MH_01_easy --out ~/datasets/euroc
```

Any KITTI sequence `00`-`21` (ground truth for `00`-`10`) or EuRoC scene (`MH_0x_*`, `V1_0x_*`,
`V2_0x_*`) works the same way; add `--dry-run` to see the download size first.

**2. Full experiment matrix (18 runs) and report tables/figures:**

```bash
KITTI_ROOT=~/datasets/kitti EUROC_ROOT=~/datasets/euroc/MH_01_easy JOBS=2 scripts/run_experiments.sh
uv run python scripts/make_qualitative_figures.py --out docs/figures \
    --config configs/kitti.yaml:$HOME/datasets/kitti:300 \
    --config configs/euroc.yaml:$HOME/datasets/euroc/MH_01_easy:1200
cd docs && tectonic mid_report.tex     # or: latexmk -pdf mid_report.tex
```

`run_experiments.sh` writes `results/<sequence>/<run>/` and regenerates `docs/tables/*` and
`docs/figures/*` (trajectories, devkit error vs. length, APE vs. distance, runtime, per-frame).

**3. One run on one scene** (the proposed configuration is the default in the configs:
sparse pipeline needs `--pipeline sparse`; `--anchor fused --refine lm` are already the YAML defaults):

```bash
# KITTI sequence 05, proposed method
uv run metricvision run configs/kitti.yaml --dataset-root ~/datasets/kitti --sequence 05 \
    --pipeline sparse --run-name sparse_fused_lm
# EuRoC V1_02_medium (the scene is chosen by its folder)
uv run metricvision run configs/euroc.yaml --dataset-root ~/datasets/euroc/V1_02_medium \
    --pipeline sparse --run-name sparse_fused_lm
# Any row of the results table: pick the pipeline, front end, anchoring and refinement
uv run metricvision run configs/kitti.yaml --dataset-root ~/datasets/kitti \
    --pipeline dense --matcher sift --refine lm --run-name dense_sift_lm
uv run metricvision run configs/kitti.yaml --dataset-root ~/datasets/kitti \
    --pipeline sparse --anchor previous --refine opencv --run-name sparse_f2f_opencv
# Quick check on the first 500 frames
uv run metricvision run configs/kitti.yaml --dataset-root ~/datasets/kitti --max-frames 500 --pipeline sparse
```

| Table row | Flags |
|---|---|
| Dense ORB+SGBM, OpenCV-LM | `--pipeline dense --matcher orb --refine opencv` |
| Dense ORB/SIFT+SGBM, our LM | `--pipeline dense --matcher orb\|sift --refine lm` |
| Sparse f2f, PnP only / OpenCV-LM / rig-LM | `--pipeline sparse --anchor previous --refine none\|opencv\|lm` |
| Sparse birth anchors, rig-LM | `--pipeline sparse --anchor birth --refine lm` |
| Sparse fused anchors, OpenCV-LM / rig-LM (proposed) | `--pipeline sparse --anchor fused --refine opencv\|lm` |

Each run prints its summary and writes `summary.json` (ATE/RPE, failures, timings),
`frame_stats.csv`, `trajectory.tum` and `trajectory.kitti` to `results/<sequence>/<run name>/`.
For the KITTI devkit metric of a single run:

```bash
uv run python -c "from pathlib import Path; from metricvision.evaluation.kitti_metrics import *; \
print(kitti_segment_errors(load_kitti_poses(Path('results/kitti_05/sparse_fused_lm/trajectory.kitti')), \
load_kitti_poses(Path.home() / 'datasets/kitti/poses/05.txt')))"
```

and for plots of a single run, the evo commands in "Running experiments" above.

## Design decisions
 
**The front end is pairwise.** `FeatureMatcher.match(view_a, view_b)` returns
2D-2D correspondences. Sparse methods implement it through
`SparseFeatureMatcher`, which caches each image's features by frame id, so a
keyframe is processed once. Dense matchers such as MASt3R have no persistent
keypoints and will implement `match` directly. Both plug into the same back end.
 
**Depth comes from fixed SGBM.** The extractor only affects keyframe-to-frame
matching and is never used for stereo matching. So each extractor is measured
in exactly one place.
 
**Tracking failure is defined explicitly.** A failure is PnP with fewer than
`pnp.min_inliers` inliers. On failure, the pose coasts on a constant-velocity
prediction, the frame becomes a new keyframe, and the failure is counted. A
pose is written for every frame, so KITTI-format files always align with
ground truth line by line.
 
**Runtime is measured honestly.** Front-end, pose and depth times are recorded
separately. `FeatureMatcher.synchronize()` is called before stopping the
front-end timer, and GPU matchers must override it with
`torch.cuda.synchronize()`. The essential-matrix inlier ratio is computed
after the timer stops, because it is a measurement, not part of the pipeline.
 
**Inlier ratio is front-end-only.** It comes from 2D-2D essential-matrix
RANSAC on the raw matches, independent of depth and PnP. It is optimistic
when the camera barely moves, so read it together with the trajectory error.
 
**Loop closure is deliberately absent,** so front-end drift stays visible.
 
**Keep these equal across extractors for fairness:** image resolution,
`max_keypoints`, ratio test settings, every `pnp.*` and `keyframe.*`
threshold, and the SGBM settings.
 
**The sparse pipeline tracks 2D features frame to frame, but anchors 3D points per track.** Each stereo pair is tracked
prev-left -> cur-left -> cur-right -> prev-right with KLT and kept only if the circle
closes within `circular_tolerance_px` and the new pair satisfies the epipolar constraint.
KLT is seeded by projecting the triangulated points with the constant-velocity prediction.
Only LM inliers stay tracked; FAST/ORB stereo detection tops up the grid when tracks fall
below `replenish_fraction * max_features` (these frames are counted as "keyframes").
`odometry.sparse_point_anchor` (or `--anchor`) selects the 3D points PnP/LM use:
`previous` re-triangulates in the previous frame (pure frame-to-frame, accumulates small biases),
`birth` keeps each track's first world-frame point, and `fused` (default) fuses every new stereo
triangulation into it by inverse covariance, so far points sharpen as they are re-observed.

**The LM is the rigid-rig estimator.** `backend/lm.py` minimises Huber-robustified,
covariance-whitened stereo reprojection error `(u_l, v_l, u_r)` over a left-multiplied
SE(3) update, with analytic Jacobians `[I | -[p]x]` and the pinhole projection Jacobian
(the right row `v_r` is left out: in a rectified rig it carries no depth and only imports rectification error).
Each residual is whitened by `J_X Sigma_X J_X^T + sigma^2 I`, where `Sigma_X` is the
triangulation covariance, so distant points (sigma_Z ~ Z^2) are down-weighted. The tests
check the Jacobians against finite differences and the covariance against Monte Carlo.

## Adding a front end
 
Pairwise front ends (dense pipeline): subclass `SparseFeatureMatcher` (or `FeatureMatcher` for dense methods),
register it, and import the module in `frontend/__init__.py`. Stereo-tracking front ends
(sparse pipeline) subclass `StereoFrontend` and register in `STEREO_FRONTENDS`:
 
```python
@MATCHERS.register("superpoint_lightglue")
class SuperPointLightGlueMatcher(SparseFeatureMatcher):
    name = "superpoint_lightglue"
 
    def extract(self, image: ImageArray) -> Features: ...
    def match_features(self, features_a: Features, features_b: Features) -> Correspondences: ...
    def synchronize(self) -> None:
        torch.cuda.synchronize()  # Required for correct GPU timings
```
 
## Checks
 
If ROS 2 is sourced in your shell, prefix these with `env -u PYTHONPATH` (ROS's pytest plugins leak into the venv otherwise).

```bash
uv run pytest            # Unit tests (Jacobians, covariance, devkit) + end-to-end synthetic runs of both pipelines
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
```
