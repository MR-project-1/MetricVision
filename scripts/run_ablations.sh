#!/usr/bin/env bash
# Method ablations of the sparse rigid-rig pipeline (one factor at a time, then the full proposal).
#
# Usage:
#   KITTI_ROOT=~/datasets/kitti EUROC_ROOT=~/datasets/euroc/MH_01_easy SEQS="00 05 07" JOBS=4 scripts/run_ablations.sh
#   KITTI_RECALIB=~/datasets/kitti_recalib/calib_{seq}.txt ...   # optional: Cvisic et al. 2021 recalibration (RQ3).
#
# Every run starts from REF = the mid-report proposal (sparse, fused xyz anchors, rigid-rig LM) with the shared
# noise model, and changes ONE thing. Folder names under results/<sequence>/:
#   Factor isolation of the rig LM (RQ1/RQ2)
#     abl_ref             REF
#     abl_mono            LM without right-image residuals (left only, same LM code)  -> value of the rigid rig
#     abl_nocov           LM without covariance whitening (pixel noise only)
#     abl_nocov_gate40    ... plus the classic 40-baseline depth gate                  -> covariance vs heuristic
#     abl_nohuber         Huber off (plain least squares)
#     abl_sigma1          LM noise pinned to 1 px (mid-report setting)                  -> shared noise model
#   Rig model (RQ3)
#     abl_independent     per-camera PnP (12 DoF), implied extrinsic deviation logged
#     abl_averaged        both per-camera estimates averaged on SE(3)
#   Method upgrades
#     abl_ransac_stereo   AP3P + MSAC on the stereo residual
#     abl_invdepth        inverse-depth anchors
#     abl_inflate         pose-uncertainty inflation of anchors
#     abl_invdepth_inflate
#     abl_adaptive        adaptive noise model
#     abl_realign5        birth-template re-alignment every 5 frames
#     abl_klt_stereo      KLT stereo matching + Shi-Tomasi score + low-threshold top-up
#     full                all upgrades together
#     full_recalib        full with the replacement KITTI calibration (only if KITTI_RECALIB is set)
set -euo pipefail

KITTI_ROOT=${KITTI_ROOT:-/data/kitti}
EUROC_ROOT=${EUROC_ROOT:-}  # Optional; empty skips EuRoC.
SEQS=${SEQS:-00}  # KITTI sequences.
JOBS=${JOBS:-3}
RESULTS=${RESULTS:-results}
KITTI_RECALIB=${KITTI_RECALIB:-}
cd "$(dirname "$0")/.."
mkdir -p "$RESULTS/logs"
unset PYTHONPATH

REF="--pipeline sparse --anchor fused --refine lm"
FULL="--set odometry.pnp.ransac=stereo --set odometry.anchor_param=inverse_depth --set odometry.anchor_pose_inflation=true \
--set odometry.noise_model=adaptive --set odometry.realign_every=5 --set stereo_frontend.stereo_matcher=klt \
--set stereo_frontend.corner_score=min_eig --set stereo_frontend.fast_threshold_min=7"
variants=(
  "abl_ref|"
  "abl_mono|--set odometry.lm_right_residuals=false"
  "abl_nocov|--set odometry.lm_covariance_weighting=false"
  "abl_nocov_gate40|--set odometry.lm_covariance_weighting=false --set odometry.max_depth_baselines=40"
  "abl_nohuber|--set odometry.pnp.lm_huber_delta=.inf"
  "abl_sigma1|--set odometry.pnp.lm_pixel_sigma=1.0"
  "abl_independent|--set odometry.rig_mode=independent"
  "abl_averaged|--set odometry.rig_mode=averaged"
  "abl_ransac_stereo|--set odometry.pnp.ransac=stereo"
  "abl_invdepth|--set odometry.anchor_param=inverse_depth"
  "abl_inflate|--set odometry.anchor_pose_inflation=true"
  "abl_invdepth_inflate|--set odometry.anchor_param=inverse_depth --set odometry.anchor_pose_inflation=true"
  "abl_adaptive|--set odometry.noise_model=adaptive"
  "abl_realign5|--set odometry.realign_every=5"
  "abl_klt_stereo|--set stereo_frontend.stereo_matcher=klt --set stereo_frontend.corner_score=min_eig --set stereo_frontend.fast_threshold_min=7"
  "full|$FULL"
)

commands=()
add_runs() {  # $1 = tag, $2 = base command
  for v in "${variants[@]}"; do
    name=${v%%|*}; extra=${v#*|}
    commands+=("$2 $REF $extra --run-name $name > $RESULTS/logs/${1}_${name}.log 2>&1")
  done
}
for seq in $SEQS; do
  base="uv run metricvision run configs/kitti.yaml --dataset-root $KITTI_ROOT --sequence $seq --output-dir $RESULTS --no-progress"
  add_runs "kitti_$seq" "$base"
  if [ -n "$KITTI_RECALIB" ]; then
    commands+=("$base $REF $FULL --set dataset.calib_file=$KITTI_RECALIB --run-name full_recalib > $RESULTS/logs/kitti_${seq}_full_recalib.log 2>&1")
  fi
done
if [ -n "$EUROC_ROOT" ]; then
  add_runs euroc "uv run metricvision run configs/euroc.yaml --dataset-root $EUROC_ROOT --output-dir $RESULTS --no-progress"
fi

printf '%s\0' "${commands[@]}" | xargs -0 -P "$JOBS" -I{} bash -c '{} && echo "done: {}" | cut -c1-160 || echo "FAILED: {}"'
