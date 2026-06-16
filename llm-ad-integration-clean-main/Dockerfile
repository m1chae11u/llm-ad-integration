FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    pkg-config \
    libcairo2-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements first to leverage Docker cache
RUN pip3 install --no-cache-dir \
    torch==2.1.0 \
    transformers==4.39.3 \
    accelerate==0.27.2 \
    bitsandbytes==0.42.0 \
    auto-gptq==0.7.1
# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONPATH=/workspace

# Command to run when container starts
CMD ["python3", "src/main.py"]