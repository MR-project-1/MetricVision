#!/usr/bin/env bash
# Run the mid-submission experiment matrix on KITTI 00 and EuRoC MH_01 (9 runs each), then build report assets.
#
# Usage (outside the container, prefix nothing; uv is called here):
#   KITTI_ROOT=~/datasets/kitti EUROC_ROOT=~/datasets/euroc/MH_01_easy JOBS=3 scripts/run_experiments.sh
#
# Runs (folder names under results/<sequence>/):
#   dense_orb_opencv  ORB + SGBM keyframe VO, PnP+RANSAC, OpenCV LM refinement   (previous baseline)
#   dense_orb_lm      ORB + SGBM keyframe VO, PnP+RANSAC, covariance-weighted LM (ours, mono residuals)
#   dense_sift_lm     SIFT + SGBM keyframe VO, PnP+RANSAC, covariance-weighted LM
#   sparse_f2f_none   FAST/ORB stereo + KLT circular tracking, frame-to-frame points, PnP+RANSAC only
#   sparse_f2f_opencv ... + OpenCV LM (left image only)
#   sparse_f2f_lm     ... + rigid-rig LM, analytic Jacobians, (u_l, v_l, u_r) residuals, covariance weighting
#   sparse_birth_lm   rigid-rig LM, 3D points anchored at each track's birth frame
#   sparse_fused_opencv / sparse_fused_lm   anchors fused by inverse covariance on every re-observation (proposal method)
<<<<<<< HEAD
#
# The LM now shares stereo_sigma_px by default (one noise model); LEGACY pins the mid-report LM noise (1 px) so
# these nine runs reproduce the mid-report numbers. The method ablations live in scripts/run_ablations.sh.
=======
>>>>>>> origin/main
set -euo pipefail

KITTI_ROOT=${KITTI_ROOT:-/data/kitti}  # Folder with sequences/ and poses/.
EUROC_ROOT=${EUROC_ROOT:-/data/euroc/MH_01_easy}  # Folder with mav0/.
JOBS=${JOBS:-3}  # Parallel runs (each is single-threaded-ish; mind RAM).
RESULTS=${RESULTS:-results}  # Results root.
cd "$(dirname "$0")/.."  # Repository root.
mkdir -p "$RESULTS/logs"  # Per-run logs.
unset PYTHONPATH  # A sourced ROS setup leaks its pytest/launch packages into the venv otherwise.

commands=()  # One shell command per run.
for dataset in kitti euroc; do
  if [ "$dataset" = kitti ]; then root=$KITTI_ROOT; else root=$EUROC_ROOT; fi
<<<<<<< HEAD
  base="uv run metricvision run configs/$dataset.yaml --dataset-root $root --output-dir $RESULTS --no-progress --set odometry.pnp.lm_pixel_sigma=1.0"  # LEGACY noise.
=======
  base="uv run metricvision run configs/$dataset.yaml --dataset-root $root --output-dir $RESULTS --no-progress"
>>>>>>> origin/main
  commands+=("$base --pipeline dense --matcher orb --refine opencv --run-name dense_orb_opencv > $RESULTS/logs/${dataset}_dense_orb_opencv.log 2>&1")
  commands+=("$base --pipeline dense --matcher orb --refine lm --run-name dense_orb_lm > $RESULTS/logs/${dataset}_dense_orb_lm.log 2>&1")
  commands+=("$base --pipeline dense --matcher sift --refine lm --run-name dense_sift_lm > $RESULTS/logs/${dataset}_dense_sift_lm.log 2>&1")
  for variant in "previous none sparse_f2f_none" "previous opencv sparse_f2f_opencv" "previous lm sparse_f2f_lm" \
                 "birth lm sparse_birth_lm" "fused opencv sparse_fused_opencv" "fused lm sparse_fused_lm"; do
    read -r anchor refine name <<< "$variant"
    commands+=("$base --pipeline sparse --anchor $anchor --refine $refine --run-name $name > $RESULTS/logs/${dataset}_${name}.log 2>&1")
  done
done

printf '%s\0' "${commands[@]}" | xargs -0 -P "$JOBS" -I{} bash -c '{} && echo "done: {}" | cut -c1-160 || echo "FAILED: {}"'  # Run in parallel.

uv run python scripts/make_report_assets.py --results "$RESULTS" --out docs \
  --config "configs/kitti.yaml:$KITTI_ROOT" --config "configs/euroc.yaml:$EUROC_ROOT"  # Tables and figures.
