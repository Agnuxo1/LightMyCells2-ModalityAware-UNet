from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]

def test_frozen_evaluation_reproduces_every_audited_result():
    runpy.run_path(str(ROOT/'tools/reproduce_audit.py'),run_name='__main__')
