# TeleChat Model Deployment Dockerfile
# This Dockerfile creates a container image for deploying the TeleChat language model

FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    git \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY requirements.txt /workspace/
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the TeleChat codebase
COPY . /workspace/

# Create directory for models
RUN mkdir -p /workspace/models

# Expose port for API service
EXPOSE 8000

# Default command
CMD ["bash"]
