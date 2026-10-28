"""Check additional evidence without changing the historical frozen manifest."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / 'source_manifest_20261027.json').read_text(encoding='utf8'))
for name, digest in manifest['files'].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
evidence = ROOT / 'evidence/ed2_20261027'
rows = list(csv.DictReader((evidence / 'supervised_holdout_metrics.csv').open(newline='', encoding='utf8')))
keys = {(r['study'], r['fov'], r['modality'], r['organelle'], r['method']) for r in rows}
assert len(rows) == len(keys) == 650
receipt = json.loads((evidence / 'supervised_holdout_receipt.json').read_text(encoding='utf8'))
assert receipt['complete'] and receipt['n_fov'] == 46 and receipt['n_studies'] == 7
audit = json.loads((evidence / 'holdout_vs_development_pixels.json').read_text(encoding='utf8'))
assert audit['complete'] and not audit['nonconstant_matches']
aggregation = json.loads((evidence / 'independent_aggregation_check.json').read_text(encoding='utf8'))
assert aggregation['passed'] and aggregation['summaries_checked'] == 40
print(json.dumps({'passed': True, 'files': len(manifest['files']), 'holdout_rows': len(rows),
                  'pixel_matches': len(audit['nonconstant_matches']), 'scope': 'local evidence, not official evaluation'}))
