FROM python:3.11-slim

# We build our own tracking server image instead of pulling the official one.
# The official MLflow image ships a cryptography version whose compiled wheel
# crashes with an illegal instruction inside the Docker Desktop virtual
# machine on Apple Silicon: the server dies at startup with no traceback.
# Versions up to 46.x work; 47.0.0 and newer crash. Keep the pin at or
# below 46.
#
# psycopg2-binary is the Postgres driver for the tracking backend. Installing
# it at build time means the container starts instantly instead of running
# pip on every boot.
RUN pip install --no-cache-dir \
      mlflow==3.15.1 \
      psycopg2-binary==2.9.12 \
      cryptography==46.0.3

EXPOSE 5000

# No CMD here on purpose: the server command, including the database
# connection string, lives in docker-compose.yml. Credentials baked into an
# image layer travel with every copy of that image.
