# Source me. Uses the main checkout's uv venv (deps identical to v2026.10.0
# except `zarrs`, which only the zarr writer tests import) with this
# worktree's fme first on the path.
export ACE_HR_ROOT=/pscratch/sd/m/mahf708/ace-hr
export FME_VENV=/pscratch/sd/m/mahf708/ace/.venv
# ACE_SRC may point at a frozen snapshot (scripts/snapshot.sh) so that edits
# to the worktree cannot leak into a running benchmark.
export PYTHONPATH=${ACE_SRC:-$ACE_HR_ROOT}${PYTHONPATH:+:$PYTHONPATH}
export PATH=$FME_VENV/bin:$PATH
export HR_DATA_CFS=/global/cfs/cdirs/e3smdata/simulations/v3.HR.F2010.aigo.PD.y1/run
export HR_DATA=$PSCRATCH/hr-data/v3.HR.F2010.aigo.PD.y1/eam.h0
export HR_EXP=$ACE_HR_ROOT/configs/experiments/2026-10-19-e3sm-hr-ace2s
# Opt-in GPU memory cap / logging for every rank (scripts/memtools). Added
# here because PYTHONPATH passed through srun --export gets reset on the
# compute nodes.
if [[ ${FME_MEMTOOLS:-0} == 1 ]]; then
  export PYTHONPATH=$HR_EXP/scripts/memtools:$PYTHONPATH
fi
