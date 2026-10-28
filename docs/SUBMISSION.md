# Grand Challenge submission

## Interface

| Item | Value |
|---|---|
| API | Grand Challenge *invoke* server (`GET /health`, `POST /invoke`), Docker label `org.grand-challenge.api-method="invoke"` |
| Input | `/input/images/organelles-transmitted-light-ome-tiff/<uuid>.tif(f)` |
| Best Focus outputs | `/output/images/{nucleus,mitochondria,tubulin,actin}-fluorescence-ome-tiff/<same name>` |
| MIP outputs | `/output/images/mip-{nucleus,mitochondria,actin,tubulin}-fluorescence-ome-tiff/<same name>` |
| Limits (mock phases) | 300 s per case, ≤ 16 GB DRAM, CPU or NVIDIA T4 16 GiB |
| Network | none: all resources are inside the image |

`src/submission/inference.py` is the Best Focus adapter. `src/submission/inference_mip_frozen.py` is the same computation with the four MIP output paths. `src/submission/model_manifest.json` lists the five checkpoints with their SHA-256.

## Building the image

The base image follows the official [demo-algorithm-template](https://github.com/DIAGNijmegen/demo-algorithm-template): `pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime`, an unprivileged `user`, the template's `app.py` server, and `numpy`, `tifffile`, `imagecodecs`, `fastapi` and `uvicorn`. The template is not vendored here. Clone it, then:

```bash
# 1. base image from the official template (adds app.py and the HTTP server)
docker build -t lmc2_base path/to/demo-algorithm-template
# 2. stage the adapter next to the model code and the five checkpoints
mkdir build && cp src/submission/inference.py src/submission/model_manifest.json \
                  src/e3/lmc_common.py src/e3/lmc_model.py build/
mkdir build/model && cp /path/to/checkpoints/fold{0,1,2,3,4}.pt build/model/
# 3. image (edit FROM in src/submission/Dockerfile to lmc2_base)
docker build -f src/submission/Dockerfile -t lmc2-bestfocus build/
# 4. self-contained variant: weights copied under /opt/app/resources/lmc2
docker build -f src/submission/Dockerfile.selfcontained -t lmc2-bestfocus-v3 build/
# 5. export for upload
docker save lmc2-bestfocus-v3 | gzip -c > lmc2-bestfocus-v3.tar.gz
```

For the MIP phase, stage `inference_mip_frozen.py` as `inference.py` and build `Dockerfile.mip-frozen` on top of the self-contained image. Verify the checkpoint hashes against `model_manifest.json` before building. The exact Python environment of the submitted image is in [container-requirements.lock.txt](container-requirements.lock.txt).

## Local acceptance tests

- `tools/check_phase_interfaces.py`: socket names and output layout for both phases.
- Contract tests (`evidence/ed2_20261027/contract_tests.json`): 13 checks for axes, metadata, error paths and TIFF resolution tags.
- Offline HTTP tests (`offline_*_receipt.json`): isolated Docker network, empty read-only `/opt/ml/model` mount, `/health` 200, `/invoke` 201, outputs identical to direct inference.
- Platform importer (`panimg_frozen_mip_receipt.json`): all output TIFFs accepted.

## Submitted artifacts

| Phase | Algorithm page | Image | Status |
|---|---|---|---|
| Best Focus – Mock | [five-fold-modality-aware-u-net-for-light-my-cells](https://grand-challenge.org/algorithms/five-fold-modality-aware-u-net-for-light-my-cells/) | `sha256:430fbab2…128a` (archive `3593e6eb…694b`) | submission `12964717-9442-47d5-8d11-092524d687fc`, evaluation **Succeeded** (28 Oct 2026) |
| MIP – Mock | [five-fold-modality-aware-u-net-mip](https://grand-challenge.org/algorithms/five-fold-modality-aware-u-net-mip/) | `sha256:b2527684…4c8e` (archive `73d5df77…e1ecd222`) | algorithm created; image prepared and locally validated |

The final test phase runs from 1 to 13 December 2026, with one submission per task.
