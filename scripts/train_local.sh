#!/bin/bash
# Local (non-HPC) runner for Isaac Lab scripts via the Apptainer sandbox.
#
# Usage:
#   ./scripts/train_local.sh --task Template-Rob6323-Go2-Biped-Direct-v0 --num_envs 2048 --headless
#
# Run a different script inside the container with SCRIPT=:
#   SCRIPT=/workspace/run/scripts/list_envs.py    ./scripts/train_local.sh
#   SCRIPT=/workspace/run/scripts/zero_agent.py   ./scripts/train_local.sh --task ... --num_envs 32 --headless
#   SCRIPT=/workspace/run/scripts/rsl_rl/play.py  ./scripts/train_local.sh --task ... --checkpoint ...
#
# Logs land in <repo>/logs/rsl_rl_local/rsl_rl/<experiment_name>/<timestamp>/

set -euo pipefail

SIF_IMAGE="${SIF_IMAGE:-/home/daniel/Dev/rl_prj/isaac-lab-base.sif}"
ISAACLAB_DIR="${ISAACLAB_DIR:-/home/daniel/Dev/rl_prj/IsaacLab}"
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_ROOT="${CACHE_ROOT:-${HOME}/.cache/isaac-local/docker-isaac-sim}"
LOGS_DIR="${RUN_DIR}/logs/rsl_rl_local"

# Isaac Sim container paths (same as train.slurm)
DOCKER_ISAACSIM_ROOT_PATH="/isaac-sim"
DOCKER_USER_HOME="/root"

# Cache dirs the container binds expect (same set train.slurm creates)
mkdir -p \
  "${CACHE_ROOT}/cache/kit" \
  "${CACHE_ROOT}/cache/ov" \
  "${CACHE_ROOT}/cache/pip" \
  "${CACHE_ROOT}/cache/glcache" \
  "${CACHE_ROOT}/cache/computecache" \
  "${CACHE_ROOT}/logs" \
  "${CACHE_ROOT}/data" \
  "${CACHE_ROOT}/documents" \
  "${CACHE_ROOT}/kit-data" \
  "${LOGS_DIR}"

# Script to run inside the container (paths are container paths under /workspace/run)
SCRIPT="${SCRIPT:-/workspace/run/scripts/rsl_rl/train.py}"

# WSL2: the CUDA stub in /usr/lib/wsl/lib talks to the GPU via /dev/dxg and the
# Windows driver store at /usr/lib/wsl/drivers; --containall hides all of them,
# so bind the whole /usr/lib/wsl tree and the device node alongside --nv.
WSL_GPU_BINDS=()
if [ -d /usr/lib/wsl ]; then
  WSL_GPU_BINDS+=(-B /usr/lib/wsl:/usr/lib/wsl:ro -B /dev/dxg)
fi

apptainer exec \
  --nv --containall \
  "${WSL_GPU_BINDS[@]}" \
  -B "${CACHE_ROOT}/kit-data:${DOCKER_ISAACSIM_ROOT_PATH}/kit/data:rw" \
  -B "${CACHE_ROOT}/cache/kit:${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache:rw" \
  -B "${CACHE_ROOT}/cache/ov:${DOCKER_USER_HOME}/.cache/ov:rw" \
  -B "${CACHE_ROOT}/cache/pip:${DOCKER_USER_HOME}/.cache/pip:rw" \
  -B "${CACHE_ROOT}/cache/glcache:${DOCKER_USER_HOME}/.cache/nvidia/GLCache:rw" \
  -B "${CACHE_ROOT}/cache/computecache:${DOCKER_USER_HOME}/.nv/ComputeCache:rw" \
  -B "${CACHE_ROOT}/logs:${DOCKER_USER_HOME}/.nvidia-omniverse/logs:rw" \
  -B "${CACHE_ROOT}/data:${DOCKER_USER_HOME}/.local/share/ov/data:rw" \
  -B "${CACHE_ROOT}/documents:${DOCKER_USER_HOME}/Documents:rw" \
  -B "${ISAACLAB_DIR}:/workspace/isaaclab:rw" \
  -B "${LOGS_DIR}:/workspace/isaaclab/logs:rw" \
  -B "${RUN_DIR}:/workspace/run:rw" \
  "${SIF_IMAGE}" bash -lc '
set -euo pipefail
cd /workspace/isaaclab
export ISAACLAB_PATH=/workspace/isaaclab
export OMNI_KIT_ACCEPT_EULA=YES
# kit often aborts at shutdown before flushing stdout — do not lose script output
export PYTHONUNBUFFERED=1
if [ -d /usr/lib/wsl/lib ]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

/isaac-sim/python.sh -m pip install --quiet -e /workspace/run/source/rob6323_go2

/isaac-sim/python.sh "$@"
' _ "${SCRIPT}" "$@"
