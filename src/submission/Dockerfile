# The existing image already includes the intact Grand Challenge app.py server.
# A fully versioned base and environment manifest are captured during packaging.
FROM lmc2_agnuxo1:latest
USER root
COPY --chown=user:user inference.py model_manifest.json lmc_common.py lmc_model.py /opt/app/
COPY --chown=user:user model/ /opt/ml/model/
USER user
ENV LMC_THREADS=1
LABEL org.grand-challenge.api-method="invoke"
ENTRYPOINT ["python", "app.py"]
