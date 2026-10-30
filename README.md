# MetricVision
Metric stereo visual odometry with a rigid-rig constraint

MetricVision estimates the metric 6-DoF motion of a calibrated stereo camera and treats the two cameras as **one
rigid body**: every stereo observation constrains a single pose through the fixed calibrated extrinsic. The project
asks three research questions:

- **RQ1 (coupling):** how much drift does the rigid-rig coupling remove compared with left-only, independent
  per-camera and loosely averaged estimation?
- **RQ2 (uncertainty):** does covariance weighting beat depth-gating heuristics and 3D-3D alignment, and is the
  predicted uncertainty statistically consistent (NEES)?
- **RQ3 (fragility):** how sensitive is the rigid constraint to extrinsic/calibration error, and can it be detected?

## The pipeline

```
stereo frame ─► circular KLT tracking (constant-velocity seeded)                      frontend/stereo_klt.py
            ─► RANSAC: OpenCV PnP or stereo AP3P-MSAC on (u_L, v_L, u_R)               backend/pose.py
            ─► rigid-rig LM: 3-D residual, covariance-whitened, Huber, pose covariance backend/lm.py
            ─► birth-template re-alignment of long tracks (drift correction)          frontend/alignment.py
            ─► landmark anchors: fused XYZ or inverse depth, pose-uncertainty inflation backend/anchors.py
            ─► sliding-window bundle adjustment over keyframes (Schur, oldest fixed)   backend/window_ba.py
            ─► adaptive noise model; replenish with ORB or KLT stereo matching         backend/stereo_odometry.py
```

Every stage is a configuration switch, so each one can be ablated in isolation (`scripts/run_ablations.sh`).
Alternative estimators for RQ2 run on the **same RANSAC inliers**: unweighted and Mahalanobis 3D-3D alignment and
SOFT-style rotation/translation decoupling (`odometry.pose_method`). Rig baselines for RQ1/RQ3 are
`odometry.rig_mode: independent | averaged`; the vertical disparity `v_L - v_R` and the implied extrinsic deviation are
logged per frame. A dense ORB/SIFT + SGBM keyframe pipeline (`pipeline: dense`) is kept **as a baseline only**.

Evaluation: evo ATE/RPE (frame-to-frame and over 1 s), the KITTI devkit metric, final/loop-closure position error,
**frame-to-frame NEES/ANEES** of the predicted pose covariance, and paired per-segment statistics (Wilcoxon signed-rank +
block-bootstrap CIs) between methods. External systems (libviso2, ORB-SLAM3, Basalt) are evaluated with the same code.

Calibration and own data: ChArUco (or checkerboard) joint stereo calibration with uncertainty, RealSense D455 rosbag
extraction (images, IMU, factory calibration) and a `stereo_folder` loader for the wheelchair sequences.

Documentation: [docs/MetricVision_Method_Upgrades.md](docs/MetricVision_Method_Upgrades.md) (every method and switch),
[docs/FINAL_DEVELOPMENT.md](docs/FINAL_DEVELOPMENT.md) (bundle adjustment, calibration, evaluation, alternatives),
[report/mid_report_revised.pdf](report/mid_report_revised.pdf).

## Project layout

```
metricvision/
├── configs/                  # kitti / euroc / synthetic / wheelchair (base) and final_* (all upgrades on), charuco_board
├── src/metricvision/
│   ├── core/                 # Types, SE(3), stereo triangulation + covariance, registry
│   ├── calibration/          # ChArUco/checkerboard detection, joint stereo calibration with uncertainty, rectification
│   ├── datasets/             # KITTI (calib override), EuRoC, stereo_folder (D455), synthetic (optional image noise)
│   ├── frontend/             # Stereo KLT front end, warped-patch re-alignment; ORB/SIFT matchers (dense baseline)
│   ├── depth/                # SGBM (dense baseline only)
│   ├── backend/              # RANSAC, rig LM, anchors, sliding-window BA, alternative estimators, sparse + dense VO
│   ├── evaluation/           # evo ATE/RPE, KITTI devkit, NEES consistency, paired statistics, writers
│   ├── config.py, cli.py     # YAML -> typed config; `metricvision run ... --set section.key=value`
│   └── pipeline.py           # Runs one experiment and writes results/<sequence>/<run>/
├── scripts/                  # run_experiments.sh, run_ablations.sh, make_report_assets.py, compare_runs.py,
│                             # evaluate_external.py, validate_methods.py, calibrate_charuco.py, calibrate_stereo.py,
│                             # make_charuco_board.py, extract_rosbag.py, download_datasets.py
├── report/                   # Revised mid report (LaTeX + PDF) and controlled-validation figures
├── docs/                     # Method documentation, generated tables and figures
└── tests/                    # 55 unit + end-to-end tests (no downloads needed)
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
 
## Datasets

```
~/datasets/euroc/MH_01_easy/mav0/...     # ASL format, extracted
~/datasets/kitti/sequences/00/...        # KITTI odometry grayscale + calibration
~/datasets/kitti/poses/00.txt            # KITTI ground truth (sequences 00-10)
~/datasets/wheelchair/lab_01/{left,right,calib.yaml}   # our D455 recordings (scripts/extract_rosbag.py)
```

```bash
uv run --with remotezip python scripts/download_datasets.py kitti --sequence 00 --out ~/datasets/kitti
uv run --with remotezip python scripts/download_datasets.py euroc --sequence MH_01_easy --out ~/datasets/euroc
```

EuRoC images are rectified on load and evaluated in the IMU body frame; KITTI is pre-rectified and evaluated in the
left camera frame. `dataset.calib_file` loads a replacement KITTI calibration (e.g. Cvisic et al. 2021, `{seq}`
placeholder) and `dataset.baseline_scale` injects a deliberate baseline error (RQ3).

## Our own data: RealSense D455

```bash
uv run python scripts/make_charuco_board.py --out docs/charuco_board_A3.pdf          # print, measure the square size
# Record with the IR emitter OFF (ros2 param set /camera/camera depth_module.emitter_enabled 0)
uv sync --extra rosbag
uv run python scripts/extract_rosbag.py --bag calib.bag --out ~/datasets/wheelchair/calib
uv run python scripts/calibrate_charuco.py --frames ~/datasets/wheelchair/calib --board configs/charuco_board.yaml \
    --out ~/datasets/wheelchair/calib.yaml --factory ~/datasets/wheelchair/calib/factory_calib.yaml
uv run python scripts/extract_rosbag.py --bag lab_01.bag --out ~/datasets/wheelchair/lab_01 --calib ~/datasets/wheelchair/calib.yaml
uv run metricvision run configs/final_wheelchair.yaml --dataset-root ~/datasets/wheelchair/lab_01 --run-name final
```

`calibrate_charuco.py --selftest` verifies the calibration on rendered views. The calibration report gives intrinsics,
baseline and extrinsic rotation with 1-sigma uncertainties and the rectified vertical error. Sequences that start and
end at the same marked spot report `loop_closure_error_m/_pct` (final translation error without ground truth).

## Running

```bash
metricvision list                                                            # Registered components
metricvision run configs/synthetic.yaml                                      # Smoke test (sparse rigid-rig VO)
metricvision run configs/kitti.yaml --dataset-root ~/datasets/kitti --sequence 05 --run-name base
metricvision run configs/final_kitti.yaml --dataset-root ~/datasets/kitti --run-name final   # all upgrades
metricvision run configs/kitti.yaml --dataset-root ~/datasets/kitti \
    --set odometry.ba_window=5 --set odometry.pnp.ransac=stereo --run-name my_variant         # any combination
```

`--set section.key=value` reaches every option (`odometry.*`, `odometry.pnp.*`, `stereo_frontend.*`, `dataset.*`, ...;
values are YAML; unknown keys are rejected). Outside a container, prefix with `uv run`.

Each run writes `results/<sequence>/<run>/`:

| File | Contents |
|---|---|
| `summary.json` | ATE, RPE (frame and 1 s), final error, loop-closure error, failures, runtimes, NEES/ANEES, vertical disparity, noise estimate, extrinsic deviation |
| `frame_stats.csv` | Per-frame matches, inliers, timings (incl. BA), pose sigma, vertical disparity, re-alignment counts |
| `pose_covariances.npz` | Per-frame 6x6 pose covariance and NEES (sparse pipeline) |
| `trajectory.tum`, `trajectory.kitti` | Trajectory in the ground-truth frame |

## Experiments for the report

```bash
# Mid-report runs (exact reproduction: LM noise pinned to 1 px)
KITTI_ROOT=~/datasets/kitti EUROC_ROOT=~/datasets/euroc/MH_01_easy JOBS=3 scripts/run_experiments.sh
# One-factor ablations + alternatives + BA + full configuration (KITTI, EuRoC, optional D455 folders)
RESULTS=results_abl SEQS="00 05 07" EUROC_ROOT=~/datasets/euroc/MH_01_easy JOBS=4 scripts/run_ablations.sh
uv run python scripts/make_report_assets.py --results results_abl --out docs_abl \
    --config "configs/kitti.yaml:$KITTI_ROOT" --config "configs/euroc.yaml:$EUROC_ROOT"
uv run python scripts/compare_runs.py --results results_abl --reference abl_ref --out docs_abl/tables \
    --config "configs/kitti.yaml:$KITTI_ROOT" --config "configs/euroc.yaml:$EUROC_ROOT"
# External baselines (run them with loop closing off, then evaluate their trajectory files)
uv run python scripts/evaluate_external.py --config configs/kitti.yaml:$KITTI_ROOT \
    --trajectory orbslam3_00.txt --format kitti --name ext_orbslam3 --results results_abl
# Controlled validation figures of the report
uv run python scripts/validate_methods.py --out report/validation
```

## Design decisions
 
**Dense baseline: the front end is pairwise.** `FeatureMatcher.match(view_a, view_b)` returns
2D-2D correspondences. Sparse methods implement it through
`SparseFeatureMatcher`, which caches each image's features by frame id, so a
keyframe is processed once. Dense matchers such as MASt3R have no persistent
keypoints and will implement `match` directly. Both plug into the same back end.
 
**Dense baseline: depth comes from fixed SGBM.** The extractor only affects keyframe-to-frame
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
 
**Bundle adjustment refines, it does not re-time.** Trajectories stay causal: each frame's pose is written when
it is estimated (BA refines the current keyframe and the landmarks; past poses are not rewritten).

**Comparisons share outlier rejection.** The alternative estimators (`pose_method`) and the rig modes reuse the same
tracks and the same RANSAC inliers, so a difference in accuracy is due to the estimator alone.

## Checks

If ROS 2 is sourced in your shell, prefix these with `env -u PYTHONPATH`.

```bash
uv run pytest            # 55 tests: Jacobians, covariance, RANSAC, anchors, BA, NEES, calibration, rosbag, end to end
uv run ruff check src scripts tests
uv run ruff format --check src tests
```
