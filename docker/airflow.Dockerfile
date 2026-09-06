FROM apache/airflow:2.10.4-python3.11

# The exercises have you run training code inside Airflow tasks, so this image
# needs the project's Python dependencies on top of Airflow itself. The pins
# match pyproject.toml so code behaves the same inside and outside Airflow.
#
# PIP_CONSTRAINT is cleared because the base image points it at Airflow's own
# constraints file, which would fight these pins.
RUN PIP_CONSTRAINT= pip install --no-cache-dir --retries 10 --timeout 120 \
      pandas==2.2.3 \
      numpy==1.26.4 \
      scikit-learn==1.5.2 \
      joblib==1.4.2 \
      mlflow==3.15.1 \
      "feast[redis]==0.49.0" \
      evidently==0.5.1 \
      prometheus-client==0.21.1 \
      pyarrow==17.0.0

# cryptography is downgraded in a separate layer on purpose. Asking pip to
# satisfy mlflow and an old cryptography in one transaction sends the resolver
# backtracking for a very long time across Airflow's dependency set; installing
# it afterwards on its own is instant.
#
# It must stay below 47 for the same illegal instruction problem described in
# mlflow.Dockerfile, and below 45 because Airflow 2.10 ships a pyopenssl that
# requires it. Airflow reads this library at startup for its Fernet key, so a
# bad version kills the scheduler with no traceback.
RUN PIP_CONSTRAINT= pip install --no-cache-dir --no-deps "cryptography==44.0.1"
