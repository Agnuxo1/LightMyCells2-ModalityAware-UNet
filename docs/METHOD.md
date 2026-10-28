# Method

## Model

The model is an equal-weight ensemble of five U-Nets. Each one has four outputs and was trained independently on one fold of a grouped five-fold split of the public Light My Cells edition-1 data.

| Component | Setting |
|---|---|
| Input | 1 normalised transmitted-light plane + 3-channel one-hot modality code (BF, PC, DIC) |
| Architecture | U-Net, 4 down-sampling levels, base width 32 (32–512 channels), two 3×3 convolutions per block, GroupNorm(8) + SiLU, transposed-convolution up-sampling, 1×1 head |
| Parameters | 7,766,372 per model |
| Outputs | 4 channels: nucleus, mitochondria, tubulin, actin |
| Loss | L1, masked per organelle so that absent labels never contribute |
| Training | 20,000 iterations per fold; seed 2026; fixed protocol (`src/e3/train_unet.py`) |
| Ensemble | arithmetic mean of the five predictions; no learned weights, no per-organelle selection |
| Precision | fp16 autocast on CUDA (T4 compatible, sm75), fp32 on CPU |

The code is in `src/e3/lmc_model.py` (network, physical rescaling) and `src/e3/lmc_common.py` (OME-TIFF reading, percentile normalisation, modality detection). The container adapter in `src/submission/inference.py` imports these same modules.

## Pre-processing

1. **Physical rescaling.** Every image is resampled to 0.2 µm/pixel from its OME `PhysicalSizeX`. Area interpolation is used for down-sampling and bilinear for up-sampling. The challenge adapter requires physical X/Y metadata, so the voxel spacing of the outputs is always valid.
2. **Intensity normalisation.** The 2nd and 99.8th percentiles (on a 4× subsample) are mapped to 0 and 1, without clipping. This matches the evaluator's own normalisation convention.
3. **Modality conditioning.** Three constant one-hot planes tell the network which microscope produced the input. One network can then learn shared morphology and modality-specific contrast.
4. **Padding.** Reflection padding to a multiple of 16, removed after prediction.

## Best Focus task

The input plane goes through the pre-processing, the five networks, the average and the resampling back to the input size. One float32 OME-TIFF is written per organelle, with the original filename, the original physical spacing and matching TIFF resolution tags (pixels/cm), tiled 128×128 with zlib compression.

## MIP task

The transmitted-light stack is normalised once, using percentiles pooled over all planes, so relative brightness between planes is preserved. Every plane is predicted by the ensemble, and the pixel-wise maximum over z is returned. The models were trained on best-focus targets, so this is an uncalibrated projection heuristic. It is evaluated against real fluorescence MIPs (see [VALIDATION.md](VALIDATION.md)).

## Robust input handling

The adapter refuses inputs it cannot interpret instead of guessing:

- more than one image series, a non-YX layout, non-singleton time or channel axes;
- missing OME metadata, an unknown modality name, missing or invalid physical pixel size;
- non-finite pixels or images smaller than 8×8.

## What is *not* in the submitted model

- No edition-2 images were used to fit or select the submitted weights.
- No third-party pretrained weights are used.
- No test-time augmentation and no post-hoc per-organelle mixing.
- Components from earlier research lines (for example the direct-MIP models in `src/mip_supervised/`) were tested and **not** promoted. See [VALIDATION.md](VALIDATION.md).
