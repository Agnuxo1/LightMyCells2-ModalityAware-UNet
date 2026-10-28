"""Check frozen evidence and publication hygiene without network, Torch or GPU."""
from pathlib import Path
import ast,csv,hashlib,json,re

ROOT=Path(__file__).resolve().parents[1]
EXPECTED='1cab58ff28a19eda0495868d96e37408ac87142e6e4292ad164465772cfadcbc'

def main():
    split=json.loads((ROOT/'splits/splits_v2.json').read_bytes())
    body={k:v for k,v in split.items() if k not in {'sha256','created_utc'}}
    canonical=json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    assert hashlib.sha256(canonical).hexdigest()==split['sha256']==EXPECTED
    groups={}
    for fov,fold in split['assign'].items():
        study=str(int(fov.split('_')[0][1:]))
        group=split['group_of_study'][study]
        groups.setdefault(group,set()).add(str(fold))
    assert len(groups)==20 and all(len(v)==1 for v in groups.values())
    manifest=json.loads((ROOT/'source_manifest.json').read_text(encoding='utf8'))
    for rel,meta in manifest['files'].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==meta['sha256'],rel
    rows=sum(sum(1 for _ in csv.DictReader(p.open(encoding='utf8',newline='')))
             for p in (ROOT/'evidence/cv_rows').glob('fold*/metrics_rows.csv'))
    assert rows==15532
    integrity=json.loads((ROOT/'evidence/integrity_verified.json').read_bytes())
    assert integrity['files']==56984 and integrity['bytes']==109287061290
    assert not integrity['errors'] and not integrity['nonfinite']
    patterns=[r'gh[pousr]_[A-Za-z0-9]{20,}',r'github_pat_[A-Za-z0-9_]{20,}',
              r'sk-(?:ant-)?[A-Za-z0-9_-]{20,}',r'https?://[^\s/@]+:[^\s/@]+@',
              r'-----BEGIN [A-Z ]*PRIVATE KEY-----']
    n=0
    for p in ROOT.rglob('*'):
        rel=p.relative_to(ROOT)
        if not p.is_file() or set(rel.parts)&{'.git','.venv','__pycache__','.pytest_cache','outputs'}:
            continue
        assert p.stat().st_size<10*1024*1024,f'Unexpected large file: {rel}'
        assert p.suffix.lower() not in {'.pt','.pth','.ckpt','.tif','.tiff','.pem','.key'},str(rel)
        assert p.name!='.env' and not p.name.startswith('.env.'),str(rel)
        if p.suffix.lower() in {'.gif','.png','.jpg','.svg'}:
            continue
        content=p.read_text(encoding='utf8')
        assert not any(re.search(pattern,content) for pattern in patterns),f'Credential pattern in {rel}; value suppressed'
        if p.suffix=='.py':ast.parse(content,filename=str(rel))
        n+=1
    print(json.dumps({'status':'passed','checked_files':n,'source_hashes':len(manifest['files']),
                      'groups_disjoint':len(groups),'evaluation_rows':rows,'raw_images_or_weights':0}))

if __name__=='__main__':main()
