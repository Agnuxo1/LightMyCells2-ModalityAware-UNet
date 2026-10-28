#!/usr/bin/env bash
# E3 — 5-fold cross-validation (splits v2, hash 1cab58ff) with TWO lanes:
# one RTX 3090 per lane (CUDA_VISIBLE_DEVICES), 2 data workers per lane (4-CPU limit).
# Usage: bash run_cv.sh <iteraciones>
set -euo pipefail
ITERS="${1:-20000}"
PROJ="$HOME/proyectos/light-my-cells-2"
PY="$PROJ/.venv/bin/python"
mkdir -p resultados

"$PY" - <<EOF
import json
from lmc_common import build_index
idx = build_index("$PROJ/meta", "$PROJ/datos", "$PROJ/splits/splits.json")
json.dump(idx, open("index.json", "w"))
from collections import Counter
print("FOV indexados:", len(idx), dict(Counter(str(e["fold"]) for e in idx.values())), flush=True)
EOF

lane() {  # $1 = physical GPU; rest = folds it runs in sequence
  local gpu="$1"; shift
  for f in "$@"; do
    echo "== GPU $gpu fold $f ($(date -u +%H:%MZ))"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" train_unet.py --index index.json --val-fold "$f" \
        --iters "$ITERS" --workers 2 --out "resultados/fold$f"
  done
}

lane 0 0 2 4 & p0=$!
lane 1 1 3 & p1=$!
rc=0
wait "$p0" || rc=1
wait "$p1" || rc=1
[ "$rc" -eq 0 ] || { echo "SOME FOLD FAILED"; exit 1; }

"$PY" summarize_cv_audit.py resultados --index index.json --out resultados/cv_audit.json
