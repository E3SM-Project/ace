# Source me. Puts the venv on PATH and this repository's fme (or a frozen copy
# in ACE_SRC) first on PYTHONPATH. Set FME_VENV if the venv is not <repo>/.venv.
_hr_scripts=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export ACE_REPO=${ACE_REPO:-$(cd "$_hr_scripts/../../../.." && pwd)}
export FME_VENV=${FME_VENV:-$ACE_REPO/.venv}
if [[ ! -x $FME_VENV/bin/python ]]; then
  echo "env.sh: no python in FME_VENV=$FME_VENV; set FME_VENV" >&2
  return 1
fi
export PATH=$FME_VENV/bin:$PATH
export PYTHONPATH=${ACE_SRC:-$ACE_REPO}${PYTHONPATH:+:$PYTHONPATH}
unset _hr_scripts
