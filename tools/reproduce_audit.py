"""Recompute the archived quality summary from its exact evaluation CSVs."""
from pathlib import Path
import json,subprocess,sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'outputs/cv_recomputed.json';out.parent.mkdir(exist_ok=True)
    cmd=[sys.executable,str(ROOT/'src/e3/summarize_cv_audit.py'),str(ROOT/'evidence/cv_rows'),
         '--index',str(ROOT/'evidence/cv_rows/index.json'),'--integrity',str(ROOT/'evidence/integrity_verified.json'),
         '--out',str(out)]
    run=subprocess.run(cmd,capture_output=True,text=True,encoding='utf8')
    if run.returncode:
        raise RuntimeError(f'Audit exit {run.returncode}: {run.stderr[:500]}')
    actual=json.loads(out.read_bytes())
    expected=json.loads((ROOT/'evidence/cv_audit_quality.json').read_bytes())
    # Only locations may differ. Every input digest and every result must agree.
    got=actual.pop('source_sha256');old=expected.pop('source_sha256')
    assert sorted(got.values())==sorted(old.values()),'Source CSV digests differ'
    assert actual==expected,'Audited results differ'
    print(json.dumps({'status':'identical_to_audited_snapshot','folds':len(actual['coverage']),
                      'rows':sum(v['observed_rows'] for v in actual['coverage'].values()),
                      'warnings':len(actual['metric_warnings'])}))

if __name__=='__main__':main()
