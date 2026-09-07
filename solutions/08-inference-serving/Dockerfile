# Reference solution for exercise 08, go further step: containerise the serving service.
#
# Build and run from the repository root:
#
#     docker build -f solutions/08-inference-serving/Dockerfile -t housing-serve .
#     docker run --rm -p 8000:8000 \
#       --network mlops-lab_default -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
#       housing-serve
#
# On the compose network the tracking server is called mlflow. From your host
# shell it is localhost:5001; inside this container it is mlflow:5000.
FROM python:3.11-slim

WORKDIR /app

# The same pins the project uses, so the model loads with the library versions
# it was trained with. cryptography stays below 47: newer wheels crash with an
# illegal instruction inside the Docker virtual machine on Apple Silicon.
RUN pip install --no-cache-dir \
      mlflow==3.15.1 \
      scikit-learn==1.5.2 \
      numpy==1.26.4 \
      pandas==2.2.3 \
      fastapi==0.115.6 \
      uvicorn==0.34.0 \
      pydantic==2.10.6 \
      prometheus-client==0.21.1 \
      cryptography==46.0.3

COPY src/__init__.py src/features.py src/serve.py ./src/

EXPOSE 8000

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
