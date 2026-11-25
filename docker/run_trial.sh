#!/bin/bash
# Run one trial (trial_runner.py) from the Docker image, by hand or from a debugger:
#   EFORECAST_WORKER_MOUNTS=$'/media/smartrue/HHD1\n/media/smartrue/HHD2' \
#     docker/run_trial.sh <path_weights> <gpu> [--refit]
#
# ``run_training.py --docker`` does NOT use this script: the orchestrator builds the whole
# ``docker run`` command as a list of arguments (eforecast/training/command_line/docker_task.py),
# which is the only way a path containing a space survives. Here the mounts are one host folder
# per line -- never a pre-quoted string, which the shell splits on whitespace without re-reading
# the quotes -- and the array below keeps each one as a single argument.
set -euo pipefail
PATH_WEIGHTS="$1"; GPU="$2"; REFIT="${3:-}"
IMAGE="${EFORECAST_WORKER_IMAGE:-eforecast-worker:latest}"
FOLDERS="${EFORECAST_WORKER_MOUNTS:-$'/media/smartrue/HHD1\n/media/smartrue/HHD2'}"

MOUNTS=()
while IFS= read -r folder; do
    [ -n "${folder}" ] && MOUNTS+=(-v "${folder}:${folder}")
done <<< "${FOLDERS}"

ARGS=(docker run --rm --gpus "device=${GPU}" --shm-size="${EFORECAST_DOCKER_SHM_SIZE:-8g}"
      --user "$(id -u):$(id -g)" -e HOME=/tmp
      -e EFORECAST_MEM_FRACTION -e PYTORCH_ALLOC_CONF
      --label "eforecast.run=${EFORECAST_RUN_ID:-manual}" --label "eforecast.pid=$$"
      "${MOUNTS[@]}"
      --entrypoint python "${IMAGE}" eforecast/training/command_line/trial_runner.py
      --path-weights "${PATH_WEIGHTS}" --gpu 0)
[ -n "${REFIT}" ] && ARGS+=("${REFIT}")
exec "${ARGS[@]}"
