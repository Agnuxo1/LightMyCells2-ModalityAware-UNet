FROM lmc2_agnuxo1:latest
USER root
COPY --chown=user:user app.py inference.py contract_io.py projection_model.py lmc_common.py lmc_model.py model_manifest.json /opt/app/
COPY --chown=user:user model/ /opt/ml/model/
USER user
ENV LMC_THREADS=1
LABEL org.grand-challenge.api-method="invoke"
ENTRYPOINT ["python", "app.py"]
