#!/usr/bin/env bash
# E1 — Light My Cells 2: download the ed.1 base dataset (S-BIAD1047, CC-BY, ~109 GB) to the server.
# The job starts in its work folder (containing run.sh and download_sbiad1047.py).
set -uo pipefail

PROJ="$HOME/proyectos/light-my-cells-2"
mkdir -p "$PROJ"
cp -f download_sbiad1047.py "$PROJ/"
cd "$PROJ"

if [ ! -x .venv/bin/python ]; then
  /usr/bin/python3 -m venv .venv
fi
.venv/bin/pip -q install --upgrade pip requests

echo "=== disco antes ==="; df -h / | tail -1
.venv/bin/python download_sbiad1047.py --accession S-BIAD1047 --out "$PROJ/datos" --meta "$PROJ/meta" --threads 4
rc=$?

echo "=== summary ==="
du -sh "$PROJ/datos" || true
echo "ficheros ome.tiff: $(find "$PROJ/datos" -name '*.ome.tiff' | wc -l)"
echo "=== disco despues ==="; df -h / | tail -1
exit $rc
