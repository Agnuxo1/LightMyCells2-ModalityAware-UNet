#!/usr/bin/env bash
# Protocol §8.3 — per-study spatial registration of TL↔fluorescence (with the partial E3 baseline, fold 0).
set -euo pipefail
PROJ="$HOME/proyectos/light-my-cells-2"
PY="$PROJ/.venv/bin/python"
mkdir -p resultados

"$PY" - <<EOF
import json
from lmc_common import build_index
idx = build_index("$PROJ/meta", "$PROJ/datos", "$PROJ/splits/splits.json")
json.dump(idx, open("index.json", "w"))
print("FOV indexados:", len(idx), flush=True)
EOF

"$PY" reg_check.py --index index.json --weights "$PROJ/models/e3_parcial_fold0.pt" \
    --per-study 12 --val-fold 0 --out resultados/reg.json
