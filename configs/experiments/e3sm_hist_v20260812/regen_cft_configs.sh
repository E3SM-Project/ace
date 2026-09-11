#!/bin/bash
# Regenerate the five committed stage-3 (-CFT) run configs from their parents'
# own config.yaml and best_inference_ckpt.tar, through make_cpl_config.py, so
# any change to the generator (2026-09-09: the coupling-interface fixes)
# reaches every run. Parents are read from the existing run yaml.
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-$D/../../../.venv/bin/python}
for f in "$D"/runs/E*-CFT*.yaml; do
  ocn=$(grep -o "/pscratch[^ ]*aug26-ft/E1[0-9]-FT[^ ]*best_inference_ckpt.tar" "$f" | head -1)
  atm=$(grep -o "/pscratch[^ ]*aug26-ft/E0[0-9]-FT[^ ]*best_inference_ckpt.tar" "$f" | head -1)
  [ -n "$ocn" ] && [ -n "$atm" ] || { echo "cannot find parents in $f"; exit 1; }
  ocn_cfg=$(dirname "$(dirname "$ocn")")/config.yaml
  atm_cfg=$(dirname "$(dirname "$atm")")/config.yaml
  # The node count sets the global batch and the inference IC count; take it
  # from the run's own .env (FME_NODES) so a B16 run at 4 nodes stays B16.
  nodes=$(sed -n 's/^FME_NODES=//p' "${f%.yaml}.env")
  [ -n "$nodes" ] || { echo "no FME_NODES in ${f%.yaml}.env"; exit 1; }
  "$PY" "$D/make_cpl_config.py" --atm-config "$atm_cfg" --ocn-config "$ocn_cfg" \
      --atm-ckpt "$atm" --ocn-ckpt "$ocn" --nodes "$nodes" --out "$f" "$@"
done
