"""Verify the published timing/equivalence records; standard library only."""
import hashlib,json,math,statistics
from pathlib import Path
root=Path(__file__).resolve().parent
r=json.loads((root/'RESULT.json').read_text())
p=json.loads((root/'PROTOCOL.json').read_text())
assert r['complete'] and r['protocol']==p
expected={(s,c['id']) for s in p['seeds'] for c in p['cases']};seen=set();pairs=[];speeds=[]
for row in r['cases']:
 key=(row['seed'],row['case']['id']);assert key in expected and key not in seen;seen.add(key)
 assert len(row['pairs'])==3 and {v['repeat'] for v in row['pairs']}=={0,1,2}
 median={n:statistics.median(v['seconds'][n] for v in row['pairs']) for n in ['baseline','streaming']}
 assert median==row['median_seconds']
 ratio=median['baseline']/median['streaming'];assert math.isclose(ratio,row['median_speedup'],rel_tol=1e-12)
 speeds.append(ratio)
 for v in row['pairs']:
  si=p['seeds'].index(row['seed']);ci=[c['id'] for c in p['cases']].index(row['case']['id'])
  assert v['order']==(['baseline','streaming'] if (si+ci+v['repeat'])%2==0 else ['streaming','baseline'])
  assert v['finite'] and v['shape']==[4,*row['case']['shape'][-2:]]
  assert v['bitwise_equal']==(v['baseline_sha256']==v['streaming_sha256'])
  pairs.append(v)
assert seen==expected and len(pairs)==27
geomean=math.exp(statistics.mean(math.log(v) for v in speeds))
assert math.isclose(geomean,r['geomean_speedup'],rel_tol=1e-12)
gate=p['promotion_gate']
checks={'nine_cases':len(seen)==9,'twentyseven_pairs':len(pairs)==27,'all_output_equivalent':all(v['allclose'] for v in pairs),
 'geomean_speedup':geomean>=gate['geomean_of_nine_case_median_speedups_at_least'],
 'no_case_regression_over_five_percent':min(speeds)>=gate['every_case_median_speedup_at_least'],
 'gpu_memory':r['gpu_peak_bytes']<gate['gpu_peak_bytes_below'],'rss_memory':r['process_peak_rss_bytes']<gate['process_peak_rss_bytes_below']}
assert checks==r['checks'] and all(checks.values())==r['candidate_promoted']
h=hashlib.sha256((root/'inference_streaming.py').read_bytes()).hexdigest()
cpu=json.loads((root/'INTEGRATION_RECEIPT.json').read_text());gpu=json.loads((root/'INTEGRATED_GPU_RECEIPT.json').read_text())
assert cpu['passed'] and gpu['passed'] and h==cpu['candidate_inference_sha256']==gpu['source_sha256']
assert len(gpu['rows'])==9 and all(v['max_abs_error']==0 and v['baseline_sha256']==v['candidate_sha256'] for v in gpu['rows'])
print(json.dumps({'passed':True,'cases':len(seen),'pairs':len(pairs),'geomean_speedup':geomean,'minimum_speedup':min(speeds),'all_tested_outputs_bitwise_equal':all(v['bitwise_equal'] for v in pairs),'deployed_source_sha256':h,'biological_improvement_proven':False,'t4_runtime_proven':False},indent=2))
