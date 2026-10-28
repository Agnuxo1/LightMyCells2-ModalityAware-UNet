# Validation protocol and results

All numbers below are local validation results. They are not official challenge scores. Each one can be recomputed from the per-image CSVs in `evidence/`.

## 1. Data

- Public edition-1 data: BioImage Archive S-BIAD1047 (CC-BY 4.0), 30 studies, 56,984 OME-TIFFs, 109,287,061,290 bytes. A full decode scan found zero decode or size errors and zero images with non-finite pixels (`evidence/integrity_verified.json`).
- Labelled best-focus fluorescence fields of view: nucleus 2,533, mitochondria 1,819, tubulin 223, actin 27. Transmitted light: BF 41,213, PC 7,670 and DIC 3,499 planes.
- One tubulin reference (Study 29, image 2542) is entirely zero. Reference-based metrics are undefined for it. It is counted in the denominator (181 valid of 182) and never imputed.

## 2. Grouped split (frozen)

- Unit: field of view. Groups: connected components of studies that share a REMBI specimen (18/22, 19/23, 20/24, 8/14/15, 9/10/13, 11/12, 5/6), which gives 20 groups. Each group sits in exactly one fold.
- Validation folds: 1,312 / 648 / 243 / 126 / 112 FOV. Fixed out-of-distribution bank: studies 1, 16, 17 and 28 (133 FOV), never used for training or selection.
- Canonical hash `1cab58ff28a19eda0495868d96e37408ac87142e6e4292ad164465772cfadcbc`. The hash is computed over the JSON without `sha256`/`created_utc`, with sorted keys, compact separators and UTF-8. `tools/check_repository.py` verifies it.

### Amendments (dated, kept as written)

1. **Amendment 1 (25 Oct 2026), partition.** The first design (v1, hash `641c64ff…`) split three very large studies into blocks across folds, which put one study in both training and validation. An independent statistical review found the leak. v1 was declared invalid, its CV run was cancelled before any result, and v2 was generated with whole groups per fold. v2 is the only partition used for reported results.
2. **Amendment 2 (25 Oct 2026), metadata and metrics.** A stale description string (`rule_ed2`) inside the frozen v2 JSON still described the v1 rule. The executed assignment is correct, and the artifact was not regenerated. The five-fold job completed all training and evaluation, but its legacy summariser stopped on a NaN standard deviation caused by the all-zero tubulin reference. Results were recovered with the corrected NaN-aware aggregator, without retraining.
3. **Amendment 3 (27 Oct 2026), official contract and new-FOV evaluation.** The official interface (one input, four separate OME-TIFF outputs) and the phase limits (300 s/case, ≤16 GB DRAM, CPU or T4) were integrated. Claims were narrowed: the split is specimen-group-disjoint, not site- or cell-line-disjoint. Study 25 provides 52.8% of validation nucleus FOVs, so equal-study estimates are reported with the equal-FOV ones. `ssim_raw` (which uses ground-truth extrema) is kept only as an oracle sensitivity, never as attainable performance.

## 3. Metrics and aggregation

- **SSIM**: `skimage.metrics.structural_similarity`, window 7, data range of the normalised ground truth. **PCC**: Pearson correlation. Prediction and ground truth are each normalised independently with float 2nd/99.8th percentiles and no clipping. On 26 October 2026 the organisers confirmed in the challenge forum that the edition-2 evaluator uses this same normalisation.
- **Independent check**: a second implementation reproduces SciPy/scikit-image on 16 raw real image pairs and synthetic pairs, with a worst error of 1.3 × 10⁻¹⁴ (`evidence/ed2_20261027/independent_raw_metrics.json`).
- **Estimand**: z planes are averaged within modality, modalities within field of view, then every FOV gets equal weight. Equal-study and per-fold views are separate estimands. The OOD bank evaluated by five models is one cohort; its spread reflects model variability, not independent samples.

## 4. Results, edition-1 grouped CV

| Organelle | FOV | Studies | SSIM (equal-FOV) | PCC (equal-FOV) | SSIM (equal-study) | Control SSIM | Control PCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nucleus | 2,441 | 26 | 0.564 ± 0.121 | 0.538 ± 0.121 | 0.666 | 0.085 | 0.041 |
| Mitochondria | 1,727 | 19 | 0.435 ± 0.128 | 0.446 ± 0.169 | 0.588 | 0.105 | 0.038 |
| Tubulin | 182 | 7 | 0.371 ± 0.139 | 0.369 ± 0.248 | 0.397 | 0.300 | 0.097 |
| Actin | 27 | 3 | 0.397 ± 0.011 | 0.378 ± 0.054 | 0.398 | 0.083 | 0.007 |

± is the SD across validation folds that contain the organelle (actin: 2 folds). Control means the transmitted-light image scored as the prediction.

**Out-of-distribution bank** (mean ± SD over the 5 fold models): nucleus SSIM 0.663 ± 0.014, PCC 0.450 ± 0.021 (92 FOV); mitochondria 0.487 ± 0.013 / 0.427 ± 0.017 (92 FOV); tubulin 0.608 ± 0.017 / 0.573 ± 0.040 (41 FOV). There is no actin in the OOD bank.

## 5. Edition-2 evaluation of the frozen ensemble

**Cohort A (development, 33 FOV, studies 77/78/82/84/86).** It was chosen before inference, using only ID disjointness from edition 1, channel coverage and size. The equal-study MIP SSIM of the unchanged ensemble is nucleus 0.706, tubulin 0.557 and actin 0.624. Mitochondria (0.498) is on par with the transmitted-light maximum projection (0.519); the interval of the difference includes zero. After these results were inspected, cohort A became training data for the direct-MIP experiment, so it is no longer a hold-out.

**Cohort B (reserved hold-out, 46 FOV, 54 FOV-modality cases, studies 76/79/80/81/83/85/87).** It was frozen before its images or outcomes were inspected. 650 metric rows were computed and checked for completeness. A pixel-exact audit found zero non-constant duplicates between the 1,268 development TIFFs and the 3,359 hold-out TIFFs.

| Organelle | FOV | Ensemble SSIM [95% CI] | Ensemble PCC | TL max SSIM | TL max PCC | TL mean SSIM | TL mean PCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nucleus | 46 | 0.623 [0.49, 0.75] | 0.725 | 0.224 | 0.123 | 0.183 | 0.042 |
| Mitochondria | 21 | 0.708 [0.66, 0.79] | 0.764 | 0.327 | 0.277 | 0.266 | 0.066 |
| Tubulin | 17 | 0.663 [0.48, 0.80] | 0.706 | 0.302 | 0.180 | 0.275 | 0.036 |
| Actin | 22 | 0.625 [0.59, 0.65] | 0.499 | 0.377 | 0.130 | 0.340 | −0.010 |

CI: 95% FOV-cluster bootstrap. Modalities are averaged within FOV first (Study 76 has both BF and PC).

## 6. Direct-MIP experiment (negative result, kept)

**Hypothesis.** A network trained directly on real fluorescence MIPs, with three transmitted-light projections as input (computed best focus, mean, max), beats the frozen max-over-planes heuristic.

**Design.** There were two fixed arms: A (masked L1, primary) and B (masked L1 + 0.2·SSIM). Both used the same initialisation (fold-0 weights), the same minibatches and augmentations, 2,000 iterations, batch 4, crop 256, AdamW 1e-4, weight decay 1e-4 and seed 20261027. They were trained on cohort A and evaluated once on cohort B. There was no early stopping and no hyper-parameter sweep.

**Outcome (equal-FOV SSIM, frozen / A / B):** nucleus 0.623 / 0.658 / 0.656, mitochondria 0.708 / 0.560 / 0.606, tubulin 0.663 / 0.598 / 0.605, actin 0.625 / 0.579 / 0.587. The nucleus gain is uncertain (its paired interval includes zero), and arm A loses on mitochondria and actin. **Decision: rejected.** The frozen ensemble stays the submission. A remote second opinion recorded with the project (JEV, `provenance=jev`) recommended keeping the frozen ensemble and stopping the bounded experiment.

## 7. Negative controls (five fixed FOV)

Each panel compares the correct model with the wrong modality code, a permuted target, a training-mean image and a blurred ground-truth oracle (`evidence/` and the delivery ledger). On mitochondria SSIM, the training-mean image (0.527) and the permuted target (0.487) score higher than the correct model (0.404) on these five FOV. This illustrates that SSIM can reward smooth images, which is why PCC is reported alongside it everywhere and why selection never relies on SSIM alone.

## 8. Engineering validation

- 13 contract tests: shapes, sockets, metadata, failure modes and TIFF resolution tags.
- Offline HTTP test of the self-contained image on an isolated Docker network, with an empty read-only `/opt/ml/model` mount: `/health` 200, `/invoke` 201, four outputs per case, maximum absolute difference to direct inference = 0.
- `panimg` (the platform's image importer) consumed all 8/8 output TIFFs at the official MIP paths.
- Runtime, RTX 3090: 2048×2048 Best Focus 2.61 s; 16-plane MIP 8.98 s; peak GPU memory 475.6 MiB.
- Platform: official Best Focus mock submission evaluated successfully on 28 October 2026.
