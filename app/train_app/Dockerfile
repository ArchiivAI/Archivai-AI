FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

# Update system packages and install git
RUN apt-get update && apt-get install -y git && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install packaging ninja
# Install flash-attention first (requires specific build environment)
RUN pip install flash-attn --no-build-isolation

# install postgresql
RUN apt-get update && apt-get install -y libpq-dev && apt-get clean && rm -rf /var/lib/apt/lists/*
# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install azureml-mlflow


# Set working directory
WORKDIR /app

COPY . /app/train_app
CMD ["uvicorn", "train_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
