# Model card · LMC2 modality-aware five-fold U-Net ensemble

| | |
|---|---|
| Task | Transmitted light → fluorescence for nucleus, mitochondria, tubulin and actin (Best Focus and MIP) |
| Inputs | One OME-TIFF, bright-field, phase contrast or DIC, 2D plane or ZYX stack, with physical pixel size |
| Outputs | Four float32 OME-TIFFs at the input size, one per organelle |
| Architecture | 5 × U-Net (7.77 M parameters each), equal-weight mean; MIP = maximum over planes |
| Training data | Public Light My Cells edition-1 images, BioImage Archive S-BIAD1047 (CC-BY 4.0) |
| Validation | Grouped 5-fold CV (20 specimen groups), fixed 4-study OOD bank, frozen edition-2 hold-out (46 FOV) |
| Licence | Apache-2.0 (code). Weights are distributed inside the challenge container only |
| Author | Francisco Angulo de Lafuente (Agnuxo1), Independent Researcher, Spain |

## Intended use

This is a research method for in-silico labelling of organelles in transmitted-light microscopy and an entry to the Light My Cells 2 challenge. Its outputs are predictions, useful for exploration, triage and method development. They are not a substitute for fluorescence imaging in quantitative biology, and they are not for clinical decisions.

## Performance summary (local, equal-FOV)

| Organelle | CV SSIM | CV PCC | Hold-out MIP SSIM | Hold-out MIP PCC |
|---|---:|---:|---:|---:|
| Nucleus | 0.564 | 0.538 | 0.623 | 0.725 |
| Mitochondria | 0.435 | 0.446 | 0.708 | 0.764 |
| Tubulin | 0.371 | 0.369 | 0.663 | 0.706 |
| Actin | 0.397 | 0.378 | 0.625 | 0.499 |

Full tables, controls and uncertainty are in [VALIDATION.md](VALIDATION.md).

## Factors and known limits

- **Data balance.** HeLa acquisitions and nucleus labels dominate. There are 27 actin FOVs from 3 studies and 182 tubulin FOVs from 7 studies, so their estimates are the least certain.
- **Independence.** Folds are specimen-group-disjoint. Laboratory and cell-line independence are not established.
- **MIP.** The projection is built from best-focus-trained predictions and is not calibrated per-plane fluorescence.
- **Metric sensitivity.** SSIM can favour smooth predictions. PCC is always reported alongside it.
- **Input contract.** Images must carry OME physical pixel size and a recognisable modality name. Otherwise the adapter stops with an explicit error.

## Hardware

fp16 on CUDA, including the NVIDIA T4 (sm75) used by the challenge, or fp32 on CPU. On an RTX 3090, a 2048² plane takes 2.6 s and a 16-plane stack 9.0 s, with a peak of 476 MiB GPU memory.
