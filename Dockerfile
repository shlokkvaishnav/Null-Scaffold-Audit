# Dockerfile for SD-MoSE
# Provides reproducible environment for training and inference

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libhdf5-dev \
    libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Julia (required for PySR)
RUN curl -fsSL https://install.julialang.org | sh -s -- -y --default-channel release
ENV PATH="/root/.juliaup/bin:${PATH}"

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PySR and Julia dependencies
RUN python -c "import pysr; pysr.install()"

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Set Python path
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Create directories for data and outputs
RUN mkdir -p data/raw data/processed checkpoints figures equations results

# Default command: run training
CMD ["python", "-m", "scripts.train.train_sdmose", "--iterations", "5"]

# Alternative commands:
# - Interactive shell: docker run -it sdmose bash
# - Preprocessing: docker run sdmose python -m scripts.data.preprocess_data
# - Visualization: docker run sdmose python -m scripts.viz.interactive_regime_map
