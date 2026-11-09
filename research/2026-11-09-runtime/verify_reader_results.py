"""Verify published correction records with Python's standard library; no GPU or data."""
import ast,hashlib,json
from pathlib import Path
r=Path(__file__).resolve().parent;d=r/'reader-fix'
def load(n):return json.loads((d/n).read_text(encoding='utf-8'))
source=(d/'inference.py').read_bytes();build=load('READER_BUILD_RECEIPT.json')
assert hashlib.sha256(source).hexdigest()==build['source_sha256']=='019be8bc00d2e6af0b2aa95e65ae255c70a24d300db1f6b9bf29d6bd148641d7'
def funcs(b):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(b).body if isinstance(n,ast.FunctionDef)}
old=funcs((r/'inference_streaming.py').read_bytes());new=funcs(source)
assert old.keys()==new.keys() and [k for k in old if old[k]!=new[k]]==['read_case']
assert build['five_weights_unchanged'] and build['only_read_case_changed']
c=load('READER_CPU_RECEIPT.json');assert c['passed'] and c['valid_real_iyx_cases']==3 and c['rejection_cases']==4
h=load('HTTP_READER_RECEIPT.json');assert h['passed'] and h['old500_reproduced'] and h['fixed201_and_four_outputs']
assert h['records'][1]['image_id']==build['image_id'] and h['records'][1]['invoke']==201 and len(h['records'][1]['outputs'])==4
g=load('READER_GPU_RECEIPT.json');assert g['passed'] and g['source_sha256']==build['source_sha256'] and len(g['rows'])==3
assert {x['modality'] for x in g['rows']}=={'BF','PC','DIC'}
for x in g['rows']:assert x['old_rejection_reproduced'] and x['prediction_bytes_equal'] and x['max_abs_error']==0 and x['input_shape']==[28,2048,2048]
assert g['no_mock_or_target_data'] and not g['t4_tested']
peer=load('independent_reader_review.json');assert all(x['passed'] for x in peer['checks'].values())
dt=load('DTYPE_OUTPUT_RECEIPT.json');assert dt['passed'] and len(dt['rows'])==6 and dt['source_sha256']==build['source_sha256']
assert {(x['dtype'],x['kind']) for x in dt['rows']}=={(d,k) for d in ['uint16','uint8'] for k in ['random','zero','max']}
for x in dt['rows']:assert x['finite'] and x['prediction_bytes_equal'] and x['serialized_four_outputs_equal'] and x['shape_and_spacing_pass']
receipt=json.loads((r/'DELIVERY_LATEST.json').read_text())
assert receipt['december_final_submitted'] is False
print(json.dumps({'PASS':True,'only_reader_changed':True,'cpu_valid':3,'cpu_rejects':4,'http_fixed_status':201,'full_shape_gpu_cases':3,'integer_tiff_output_cases':6,'official_evaluation_status':receipt['evaluation_status'],'local_checks_do_not_establish_official_status':True}))
