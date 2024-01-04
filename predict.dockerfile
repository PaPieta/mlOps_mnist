# Base image
FROM python:3.10-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mlOps_mnist/ mlOps_mnist/
COPY data/ data/
COPY models/ models/


WORKDIR /
RUN pip install . --no-cache-dir #(1)

ENTRYPOINT ["python", "-u", "mlOps_mnist/models/predict_model.py", "-model_path=models/test/model.pth", "-data_path=data/raw/corruptmnist/test_images.pt"]