#!/usr/bin/env bash
# E3 — UNet-L1 baseline. Usage: bash run.sh <iters> <val_fold> <max_eval_fov(0=all)>
set -euo pipefail
ITERS="${1:-20000}"; FOLD="${2:-0}"; MAXEV="${3:-0}"
PROJ="$HOME/proyectos/light-my-cells-2"
PY="$PROJ/.venv/bin/python"
"$PY" -m pip -q install numpy tifffile scikit-image scipy torch imagecodecs
"$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
"$PY" - <<EOF
import json
from lmc_common import build_index
idx = build_index("$PROJ/meta", "$PROJ/datos", "$PROJ/splits/splits.json")
json.dump(idx, open("index.json", "w"))
from collections import Counter
print("FOV indexados:", len(idx), Counter(str(e["fold"]) for e in idx.values()))
EOF
mkdir -p resultados
"$PY" train_unet.py --index index.json --val-fold "$FOLD" --iters "$ITERS" --max-eval-fov "$MAXEV" --out resultados
"$PY" summarize.py resultados/metrics_rows.csv resultados/resumen.json
cp index.json resultados/
