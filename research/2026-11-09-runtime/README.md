# Frozen MIP transport and reader update

**Francisco Angulo de Lafuente** · ORCID [0009-0001-1634-7063](https://orcid.org/0009-0001-1634-7063) · 9 November 2026.

Keeping the maximum over z on the GPU removes one full-resolution device-to-host copy per plane. The five trained models, normalization, interpolation, output geometry and organelle sockets remain fixed. This changes transport, with no fitting or ground-truth use.

A preregistered comparison used three fixed synthetic case types, three seeds and three paired repetitions: nine cases and 27 pairs. The geometric mean throughput ratio was **1.09472x**; the minimum case ratio was 0.99621x. All 27 paired outputs had matching bytes and SHA-256. On the 16-plane 2048 x 2048 case, median time fell from 18.51–18.61s to 16.14–16.21s. These are RTX 3090 / host PyTorch 2.14.1 measurements, not T4 timings or biological accuracy improvements.

The transport-only adapter also produced matching bytes in nine GPU cases. Its image's own PyTorch 2.13 CPU environment passed three input/output cases and two invalid-input rejections. The first official mock then exposed a reader gap: a single-channel z-stack reported as IYX was rejected before inference. A scoped reader correction, its reproduction, actual-image CPU/API checks, full-shape GPU checks and independent review are preserved in [reader-fix](reader-fix/README.md). The model, weights and prediction functions remain fixed. [Latest official status](DELIVERY_LATEST.json) and [delivery history](DELIVERY_HISTORY.json) distinguish reception, execution and results.


```mermaid
flowchart LR
 A[TL stack and metadata] --> B[Fixed preprocessing]
 B --> C[Five frozen models per plane]
 C --> D[Mean and resize]
 D --> E[Maximum over z on GPU]
 E --> F[One transfer to CPU]
 F --> G[Four MIP OME-TIFF outputs]
```

## Verify the records

Run `python verify_results.py` and `python verify_reader_results.py`. These standard-library checks reconstruct pairing, medians and the fixed promotion gate, cross source hashes with CPU/GPU receipts, and verify the constrained reader evidence. The records contain output digests rather than full prediction arrays. Re-execution requires the source, original five weights and the declared runtime. Local checks alone do not establish official execution, score or rank.

## Rebuild and rerun

`inference_streaming.py` preserves the exact first transport-only adapter source. The corrected selected adapter is `reader-fix/inference.py`. Both Dockerfile stages overlay only `/opt/app/inference.py`; [REBUILD_FROM_BACKUP.md](REBUILD_FROM_BACKUP.md) gives the complete tag chain, checkpoint verification and runtime limitations. The original five weights are backed up on Hugging Face and are not stored in Git.

For the native comparison, prepare a baseline directory containing the original MIP adapter as `inference.py`, unchanged `lmc_common.py`, `lmc_model.py` and `model_manifest.json`. Set `LMC_BASELINE_ROOT` to it, `LMC_MODEL_DIR` to the five-weight directory, `LMC_STUDY_OUTPUT` to a fresh output directory, `LMC_DEVICE=cuda:0`, and expose exactly one GPU. Run `study.py` under the stated project runtime and resource budget. `streaming.py` is the unchanged prototype used by that study.

The first two launch attempts failed before inference because the server had no Docker GPU driver and its host environment lacked tifffile. The recovered native study used a project-local tifffile 2026.10.15. An independently reproduced signed-zero diagnostic defect was corrected before the successful run; numerical candidates and promotion criteria were unchanged. The smallest case has essentially unchanged latency. No T4, SSIM/PCC, clinical, generalization or ranking benefit follows from these timing checks.

Original project code and this update retain Apache-2.0. Public microscopy data retain their own CC-BY attribution requirements. This research directory contains source and result records, with no microscopy images, targets or credentials. The complete repository preserves the original credited GIF figures, including rendered public microscopy and ground-truth crops; see `docs/THIRD_PARTY.md`. Raw/full-resolution datasets and third-party runtime binaries are excluded from the backups.
