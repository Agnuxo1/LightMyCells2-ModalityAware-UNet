"""Ensure the two saved adapters differ only by their official output paths."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
base=(root/'src/submission/inference.py').read_text(encoding='utf8')
mip=(root/'src/submission/inference_mip_frozen.py').read_text(encoding='utf8')
old="OUTPUTS = {o: f'images/{o.lower()}-fluorescence-ome-tiff' for o in ORGS}"
new="OUTPUTS = {o: f'images/mip-{o.lower()}-fluorescence-ome-tiff' for o in ORGS}"
assert base.count(old)==1 and base.replace(old,new)==mip
evidence=root/'evidence/ed2_20261027'
offline=json.loads((evidence/'offline_frozen_mip_receipt.json').read_text(encoding='utf8'))
panimg=json.loads((evidence/'panimg_frozen_mip_receipt.json').read_text(encoding='utf8'))
assert offline['passed'] and len(offline['cases'])==2
assert all(case['max_cpu_direct_error']==0 for case in offline['cases'])
expected={f'mip-{x}-fluorescence-ome-tiff' for x in ('nucleus','mitochondria','actin','tubulin')}
assert panimg['passed'] and {r['socket'] for r in panimg['records']}==expected
print(json.dumps({'passed':True,'identical_except_output_paths':True,'cases':2,'outputs_read_by_panimg':panimg['n_consumed_files']}))
