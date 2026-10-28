# Reproducing

## 1. Repository audit (CPU, no data, under a minute)

```bash
python -m pip install -r requirements-dev.txt
python tools/check_repository.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python tools/reproduce_audit.py
```

- `check_repository.py` verifies the canonical split hash, that the 20 specimen groups are disjoint across folds, the hashes in `source_manifest.json`, the 15,532 evaluation rows and the integrity record. It also scans for credentials, weights and raw images.
- `reproduce_audit.py` rebuilds the complete cross-validation summary from the per-image CSVs and compares every value with the archived `evidence/cv_audit_quality.json`.

## 2. Data

Download the public edition-1 data (S-BIAD1047, CC-BY 4.0, 109.3 GB) with the resumable downloader:

```bash
python src/download_sbiad1047.py --accession S-BIAD1047 --out data --meta meta --threads 4
```

The edition-2 challenge data are available to registered participants from the challenge [database page](https://lightmycells2.grand-challenge.org/database/).

## 3. Training the five folds

```bash
python -m pip install -r requirements.txt
# one fold: <iterations> <fold> <max_eval_fov>
bash src/e3/run.sh 20000 0 0
# all folds, two GPUs
bash src/e3/run_cv.sh
```

The runners contain the absolute paths of the original server workspace. Point `ROOT`/data paths at your copy before running them. The split is read from `splits/splits_v2.json`. Never regenerate it for reported results. The checkpoint digests of the submitted ensemble are in `evidence/checkpoints_manifest.json` and `src/submission/model_manifest.json`.

Environment locks:

- Training/evaluation: `evidence/ed2_20261027/training-requirements.lock.txt`
- Container: `docs/container-requirements.lock.txt`

## 4. Evaluation summaries

```bash
python src/e3/summarize_cv_audit.py --help
```

The aggregator averages planes within modality, modalities within FOV, then gives every FOV equal weight. It reports equal-study and per-fold views. It treats constant references as undefined and keeps them in the denominators.

## 5. README figures

```bash
# on a machine with the data and checkpoints (CPU is enough)
python tools/make_showcase_arrays.py <index.json> <model_dir> <out_dir> S24_390 S25_1031 S20_308
# anywhere
python figures/make_readme_gifs.py --showcase <out_dir>
```

Each showcase FOV is predicted by the fold model whose validation fold contains it, so every image panel is an out-of-fold prediction.

## 6. Direct-MIP experiment (negative result)

`src/mip_supervised/` holds the cache builder, training script, hold-out evaluator and adapter used for the rejected direct-MIP arms. The development cohort (studies 77/78/82/84/86) and the reserved hold-out (studies 76/79/80/81/83/85/87) are listed in `evidence/ed2_20261027/cohort_frozen.json` and `holdout_frozen.json`.

## Provenance note

The files in `evidence/executed_source/` are byte-exact copies of the scripts that produced the archived metrics. The public versions in `src/` differ only in comments and docstrings, which were translated to English. Their abstract syntax trees are identical, and the mapping is in `source_manifest.json`.
