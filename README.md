# MetricVision
Feature extractors for visual odometry, measured end to end

Benchmark visual odometry front ends (feature extraction + matching) inside one
fixed stereo VO back end, so trajectory differences come from the extractor and
nothing else.
 
Status: ORB and SIFT front ends, SGBM stereo depth, PnP + RANSAC keyframe
tracking, and evaluation with evo, on EuRoC, KITTI and a built-in synthetic
sequence for smoke testing. SuperPoint+LightGlue, ALIKED+LightGlue, MASt3R, sliding-window
bundle adjustment and the ROS 2 node come next.
 
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
├── src/vobench/
│   ├── core/                 # Data types, SE(3) helpers, component registry
│   ├── datasets/             # StereoDataset ABC + EuRoC, KITTI, synthetic loaders
│   ├── frontend/             # FeatureMatcher ABC + ORB, SIFT  <- the variable under test
│   ├── depth/                # DepthEstimator ABC + SGBM       <- fixed
│   ├── backend/              # PnP, keyframe policy, VO loop    <- fixed
│   ├── evaluation/           # Front-end metrics, ATE/RPE via evo, output writers
│   ├── config.py             # YAML -> typed config
│   ├── pipeline.py           # Runs one (sequence, front end) experiment
│   └── cli.py                # `vobench run` / `vobench list`
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
2. Run **Dev Containers: Reopen in Container** and choose **vobench (laptop, CUDA)**.
3. The container runs `uv sync --all-groups` on creation. Check with
   `vobench run configs/synthetic.yaml`.
   
### Option C: Jetson dev container
 
1. On the Jetson, run `cat /etc/nv_tegra_release`. Set `L4T_TAG` in
   `.devcontainer/jetson/devcontainer.json` to match (e.g. `R36 … REVISION: 4.0` → `r36.4.0`).
2. From the laptop, use **Remote-SSH** to connect to the Jetson and open the cloned repo.
3. Run **Reopen in Container** and choose **vobench (Jetson AGX)**.
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
```
 
Outside a container, prefix commands with `uv run`. `--matcher` keeps the
shared parameters (`max_keypoints`, `ratio`, `mutual_check`), so the keypoint
budget stays equal when you switch extractors.
 
Each run writes to `results/<sequence>/<matcher>/`:
 
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
 
## Adding a front end
 
Subclass `SparseFeatureMatcher` (or `FeatureMatcher` for dense methods),
register it, and import the module in `frontend/__init__.py`:
 
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
 
```bash
uv run pytest            # Unit tests + end-to-end synthetic runs (ORB and SIFT)
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
```
 
## Roadmap
 
1. Run ORB and SIFT on EuRoC MH_01–05 and V1/V2, and KITTI 00–10, then tune shared thresholds once.
2. Add SuperPoint+LightGlue and ALIKED+LightGlue (PyTorch; NVIDIA's wheels on the Jetson).
3. Add MASt3R as a dense `FeatureMatcher`, using its 2D matches only and never its 3D output.
4. Add sliding-window bundle adjustment as a switchable back-end stage.
5. Add a perturbation layer (motion blur, low light with sensor noise, texture removal) with severity sweeps.
6. Add a ROS 2 node wrapping `StereoVisualOdometry` for the D455 and live runs on the robot.