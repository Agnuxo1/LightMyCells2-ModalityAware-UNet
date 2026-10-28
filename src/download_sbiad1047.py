#!/usr/bin/env python3
"""Download S-BIAD1047 (Light My Cells ed.1, CC-BY) from BioStudies/EBI.

Resumable: skips files already present with the exact size. Progress log
in <meta>/manifest.jsonl and final summary in <meta>/resumen.json.
Usage: python download_sbiad1047.py [--accession S-BIAD1047] [--out datos] [--meta meta] [--threads 4]
"""
import argparse
import concurrent.futures as cf
import json
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://www.ebi.ac.uk/biostudies/files/{acc}/{rel}"
UA = {"User-Agent": "lmc2-sprint-downloader/1.0 (research; contact: fran concursos-24h)"}


def fetch_json(url: str, retries: int = 4):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            if i == retries - 1:
                raise
            print(f"retry {i+1} {url}: {e}", flush=True)
            time.sleep(5 * (i + 1))


def download_one(args):
    acc, rel, size, out_root, meta_dir = args
    dest = out_root / rel
    if dest.exists() and dest.stat().st_size == size:
        return ("skip", rel, size, 0.0)
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = BASE.format(acc=acc, rel=rel)
    t0 = time.time()
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers=UA)
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            if tmp.stat().st_size != size:
                tmp.unlink(missing_ok=True)
                raise IOError(f"size mismatch {rel}: {tmp.stat().st_size} != {size}")
            tmp.rename(dest)
            return ("ok", rel, size, time.time() - t0)
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                return ("error", rel, size, str(e)[:200])
            time.sleep(10 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accession", default="S-BIAD1047")
    ap.add_argument("--out", default="datos")
    ap.add_argument("--meta", default="meta")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    out_root = Path(a.out).resolve()
    meta = Path(a.meta).resolve()
    meta.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    manifest = fetch_json(BASE.format(acc=a.accession, rel=f"{a.accession}.json"))
    (meta / "study_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    import re
    fls = sorted(set(re.findall(r'FileList/[^"\']+\.json', json.dumps(manifest))))
    print(f"FileLists: {len(fls)}", flush=True)

    tasks, per_study = [], {}
    for fl in fls:
        entries = fetch_json(BASE.format(acc=a.accession, rel=fl))
        (meta / (fl.split("/")[-1])).write_text(json.dumps(entries), encoding="utf-8")
        n = 0
        for e in entries:
            if e.get("type") == "file" and "image_" in str(e.get("path", "")):
                tasks.append((a.accession, e["path"], int(e.get("size", 0)), out_root, meta))
                n += 1
        per_study[fl.split("/")[-1].replace(".json", "")] = {
            "ficheros": n, "bytes": sum(t[2] for t in tasks[-n:]) if n else 0}
    total_bytes = sum(t[2] for t in tasks)
    print(f"TOTAL: {len(tasks)} ficheros, {total_bytes/1e9:.2f} GB", flush=True)

    done = skipped = errors = 0
    got_bytes = 0
    log = open(meta / "manifest.jsonl", "a", encoding="utf-8")
    with cf.ThreadPoolExecutor(a.threads) as ex:
        for status, rel, size, info in ex.map(download_one, tasks):
            if status == "skip":
                skipped += 1
            elif status == "ok":
                done += 1
                got_bytes += size
                log.write(json.dumps({"f": rel, "b": size, "s": round(info, 2)}) + "\n")
                log.flush()
            else:
                errors += 1
                log.write(json.dumps({"error": rel, "detail": str(info)}) + "\n")
                log.flush()
            if (done + skipped + errors) % 200 == 0:
                el = time.time() - t_start
                print(f"prog: ok={done} skip={skipped} err={errors} "
                      f"descargado={got_bytes/1e9:.1f}GB elapsed={el/60:.1f}min", flush=True)
    log.close()

    resumen = {
        "accession": a.accession,
        "ficheros_total": len(tasks), "ok": done, "skip": skipped, "errors": errors,
        "gb_descargados": round(got_bytes / 1e9, 2),
        "por_estudio": per_study,
        "minutos": round((time.time() - t_start) / 60, 1),
    }
    (meta / "resumen.json").write_text(json.dumps(resumen, indent=1), encoding="utf-8")
    print(json.dumps(resumen, indent=1), flush=True)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
