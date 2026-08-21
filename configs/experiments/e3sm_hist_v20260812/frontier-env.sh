#!/usr/bin/env bash
# Login/build-time environment for ACE (fme) on OLCF Frontier.
#
#   source configs/experiments/e3sm_hist_v20260812/frontier-env.sh
#
# Stack: ROCm 7.1.1 + PyTorch 2.10.0+rocm7.1, Python 3.12 (uv-managed), venv at $ACE_ROOT/.venv
# Hardware: AMD MI250X, gfx90a. 4 MI250X/node = 8 GCDs; Slurm and ROCr treat each GCD as one GPU.
#
# See NOTES-frontier-env.md for how to build the venv this activates.
# Refs: https://docs.olcf.ornl.gov/software/analytics/pytorch_frontier.html
#       https://docs.olcf.ornl.gov/systems/frontier_user_guide.html

# Repo root, inferred from this script's location unless overridden.
if [ -z "${ACE_ROOT}" ]; then
    _this="${BASH_SOURCE[0]:-$0}"
    ACE_ROOT="$(cd "$(dirname "${_this}")/../../.." && pwd)"
fi
export ACE_ROOT

module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load rocm/7.1.1
module load craype-accel-amd-gfx90a

# Cray runtime libs (libmpi_gtl_hsa, libfabric, ...) must be resolvable at runtime.
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"

# GPU-aware Cray MPICH.
export MPICH_GPU_SUPPORT_ENABLED=1

# Keep uv's cache off $HOME (NFS, small quota).
export UV_CACHE_DIR="${UV_CACHE_DIR:-/lustre/orion/cli115/proj-shared/${USER}/.uv-cache}"

if [ -d "${ACE_ROOT}/.venv" ]; then
    source "${ACE_ROOT}/.venv/bin/activate"
else
    echo "note: ${ACE_ROOT}/.venv does not exist yet -- see NOTES-frontier-env.md" >&2
fi
