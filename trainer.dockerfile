# Base image
# FROM python:3.10-slim
FROM  nvcr.io/nvidia/pytorch:22.07-py3


# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mlOps_mnist/ mlOps_mnist/
COPY data/ data/
COPY models/ models/

# If nvidia go to workspace
WORKDIR /workspace/
# WORKDIR /
# RUN ls
# RUN pwd
RUN pip install . --no-cache-dir #(1)

ENTRYPOINT ["python", "-u", "mlOps_mnist/models/train_model.py"]
