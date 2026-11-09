# Reconstructing the frozen adapter from source and checkpoint backups

These steps clarify the historical Docker tag dependencies without changing the original Dockerfiles. They reconstruct the supplied source with the five original checkpoints; reproducing an identical Docker image ID additionally requires the exact original base layers. The public backup contains source and own weights, not the Docker base image or third-party runtime binaries.

Start at the repository root on Linux with Docker. Extract the five checkpoints from the Hugging Face model archive into `/path/to/checkpoints`. Compare every file against `weights_manifest.json` and `src/submission/model_manifest.json` before building.

Prepare a compatible `lmc2_base` from the official [Grand Challenge algorithm template](https://github.com/DIAGNijmegen/demo-algorithm-template), as described in `docs/SUBMISSION.md`. It must provide the template `app.py` invoke server at `/opt/app`, the unprivileged `user`, and the Python packages in `docs/container-requirements.lock.txt`. The submitted runtime used PyTorch2.13.0/CUDA12.6. A different upstream template or dependency version needs renewed HTTP and TIFF acceptance checks; an arbitrary template build is not proof of equivalent behavior.

```bash
docker tag lmc2_base lmc2_agnuxo1:latest
mkdir -p build/model
cp src/submission/inference.py src/submission/model_manifest.json build/
cp src/e3/lmc_common.py src/e3/lmc_model.py build/
cp /path/to/checkpoints/fold{0,1,2,3,4}.pt build/model/

# Exact tag names consumed by the original FROM statements.
docker build -f src/submission/Dockerfile -t lmc2-official-20261027-v2 build/
docker build -f src/submission/Dockerfile.selfcontained -t lmc2-selfcontained-20261027-v3 build/
cp src/submission/inference_mip_frozen.py build/inference.py
docker build -f src/submission/Dockerfile.mip-frozen -t lmc2-frozen-mip-20261027-v1 build/

cp research/2026-11-09-runtime/inference_streaming.py build/inference.py
docker build -f research/2026-11-09-runtime/Dockerfile -t lmc2-mip-streaming-20261109-v1 build/
cp research/2026-11-09-runtime/reader-fix/inference.py build/inference.py
docker build -f research/2026-11-09-runtime/reader-fix/Dockerfile -t lmc2-mip-iyx-20261109-v2 build/
docker save lmc2-mip-iyx-20261109-v2 | gzip -c > lmc2-mip-iyx-20261109-v2.tar.gz
```

Run `python tools/check_repository.py`, `python research/2026-11-09-runtime/verify_results.py` and `python research/2026-11-09-runtime/verify_reader_results.py` to verify published records without GPU or data. These commands check evidence consistency, not newly built container execution. Before any new platform submission, test the built container offline with its default image user, `GET /health`, `POST /invoke`, and all four correctly named, finite OME-TIFF outputs. Follow the input/output contract in `docs/SUBMISSION.md` and retain the newly built image and archive hashes.

The preserved executed reader test scripts assume the original server project paths. The CPU reader check additionally expects `/old/inference.py` to be the previous streaming adapter and a writable synthetic receipt directory. Configure those fixture paths explicitly when rerunning elsewhere. Native GPU results used RTX3090/PyTorch2.14.1; they do not replace testing the image runtime or the official T4 mock.
