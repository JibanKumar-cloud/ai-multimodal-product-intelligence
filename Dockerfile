FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-venv git wget curl \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Install project in development mode
RUN pip3 install -e .

# Default: open a shell
CMD ["/bin/bash"]

# Example runs:
# Training:   docker run --gpus all -v $(pwd)/data:/app/data wayfair-catalog-ai python3 scripts/train.py
# Inference:  docker run --gpus all wayfair-catalog-ai python3 scripts/run_inference.py --model rule-based
# Demo:       docker run --gpus all -p 8501:8501 wayfair-catalog-ai streamlit run demo/app.py --server.address 0.0.0.0
