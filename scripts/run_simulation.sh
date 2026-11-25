#!/usr/bin/env bash
# Offline evaluation: both approaches on every recorded wheelchair bag and on KITTI (EuRoC too if present),
# then the comparison report (results/simulation/report.html, report.md, plots/).
#
#   classical  FAST + ORB stereo matching, KLT tracking        -> rigid-rig LM + sliding-window rigid-rig BA (5 KF)
#   learned    SuperPoint + LightGlue (GPU), same rig checks   -> the same rigid-rig LM + window BA
#
# Usage (repository root, inside the dev container or with `uv sync --all-extras` on the host):
#   scripts/run_simulation.sh
#   BAGS_DIR=ros_bags KITTI_ROOT=/data/kitti KITTI_SEQ=00 EUROC_ROOT=/data/euroc/MH_01_easy JOBS=2 scripts/run_simulation.sh
#   MAX_FRAMES=300 scripts/run_simulation.sh   # quick check
set -euo pipefail
cd "$(dirname "$0")/.."
unset PYTHONPATH  # A sourced ROS setup would leak its packages into the uv environment.

BAGS_DIR=${BAGS_DIR:-$([ -d ros_bags ] && echo ros_bags || echo /data/wheelchair_bags)}  # Bag folders (metadata.yaml + .db3).
KITTI_ROOT=${KITTI_ROOT:-$([ -d /data/kitti ] && echo /data/kitti || echo "$HOME/datasets/kitti")}
KITTI_SEQ=${KITTI_SEQ:-00}
EUROC_ROOT=${EUROC_ROOT:-}
RESULTS=${RESULTS:-results/simulation}
JOBS=${JOBS:-2}  # The learned runs use the GPU, the classical ones the CPU: two in parallel overlap well.
MAX_FRAMES=${MAX_FRAMES:-}

mkdir -p "$RESULTS/logs"
extra=${MAX_FRAMES:+--max-frames $MAX_FRAMES}
commands=()
add() {  # add <config> <dataset root> <log prefix> [extra args]
  local config=$1 root=$2 name=$3; shift 3
  for approach in "classical fast_orb_klt" "learned superpoint_lightglue"; do
    read -r run frontend <<< "$approach"
    commands+=("uv run metricvision run $config --dataset-root $root $* --pipeline sparse --window 5 --stereo-frontend $frontend \
--run-name $run --output-dir $RESULTS --no-progress $extra > $RESULTS/logs/${name}_${run}.log 2>&1 \
&& echo 'done:   ${name} ${run}' || echo 'FAILED: ${name} ${run} (see $RESULTS/logs/${name}_${run}.log)'")
  done
}

for bag in "$BAGS_DIR"/*/; do
  bag=${bag%/}
  [ -f "$bag/metadata.yaml" ] && add configs/wheelchair.yaml "$bag" "wheelchair_$(basename "$bag")"
done
[ -d "$KITTI_ROOT/sequences/$KITTI_SEQ" ] && add configs/kitti.yaml "$KITTI_ROOT" "kitti_$KITTI_SEQ" --sequence "$KITTI_SEQ"
[ -n "$EUROC_ROOT" ] && [ -d "$EUROC_ROOT/mav0" ] && add configs/euroc.yaml "$EUROC_ROOT" "euroc_$(basename "$EUROC_ROOT")"

echo "running ${#commands[@]} runs ($JOBS in parallel), logs in $RESULTS/logs"
printf '%s\0' "${commands[@]}" | xargs -0 -P "$JOBS" -I{} bash -c '{}'
uv run python scripts/simulation_report.py --results "$RESULTS"
