# Third-party resources and attribution

| Resource | Use | Licence / terms |
|---|---|---|
| Light My Cells edition-1 images, BioImage Archive [S-BIAD1047](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD1047), France-BioImaging | training and cross-validation; small crops rendered in `assets/` | CC-BY 4.0 (attribution) |
| Light My Cells 2 challenge data ([database](https://lightmycells2.grand-challenge.org/database/)) | local evaluation only; not redistributed | challenge rules, CC-BY |
| [DIAGNijmegen/demo-algorithm-template](https://github.com/DIAGNijmegen/demo-algorithm-template) | base container and `app.py` invoke server (not vendored) | see the template repository |
| [panimg](https://github.com/DIAGNijmegen/panimg) | local import check of output TIFFs | Apache-2.0 |
| PyTorch, NumPy, SciPy, scikit-image, tifffile, imagecodecs, FastAPI, Uvicorn | runtime and evaluation | their respective open-source licences |

No third-party source code or pretrained weights are vendored in this repository. BiaPy and the published first-edition methods (VQGAN-based and U-Net-based entries) were studied as literature. This implementation is independent and does not reproduce them.

AI assistants (Anthropic Claude, OpenAI Codex) helped with code, review and documentation under the author's direction. Every reported number comes from archived, executed artifacts in `evidence/`.

Please cite the Light My Cells challenge and dataset publications when they appear, in line with the challenge rules.
