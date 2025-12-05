#!/bin/bash
# build-and-publish.sh - Local Docker build and publish script
# Usage: ./build-and-publish.sh [cpu|gpu]

set -e

# Fail early if credentials are missing
if [ -z "$DOCKERHUB_USERNAME" ] || [ -z "$DOCKERHUB_TOKEN" ]; then
    echo "ERROR: DOCKERHUB_USERNAME and DOCKERHUB_TOKEN environment variables are required"
    echo "Please set them before running this script:"
    echo "  export DOCKERHUB_USERNAME=your-username"
    echo "  export DOCKERHUB_TOKEN=your-token"
    exit 1
fi

# Parse variant argument
VARIANT="${1:-cpu}"
if [ "$VARIANT" != "cpu" ] && [ "$VARIANT" != "gpu" ]; then
    echo "ERROR: Invalid variant. Must be 'cpu' or 'gpu'"
    echo "Usage: $0 [cpu|gpu]"
    exit 1
fi

# Configuration
IMAGE_NAME="${DOCKERHUB_USERNAME}/aichi2lm"
TAG="${TAG:-latest}"
IMAGE_TAG="${IMAGE_NAME}:${TAG}-${VARIANT}"

echo "Building Docker image: ${IMAGE_TAG}"

# Login to Docker Hub
echo "Logging in to Docker Hub..."
echo "$DOCKERHUB_TOKEN" | docker login --username "$DOCKERHUB_USERNAME" --password-stdin

# Build the image
if [ "$VARIANT" = "gpu" ]; then
    echo "Building GPU variant..."
    docker build -f Dockerfile.full-gpu -t "$IMAGE_TAG" .
else
    echo "Building CPU variant..."
    docker build -f Dockerfile.full-cpu -t "$IMAGE_TAG" .
fi

# Tag as latest for this variant
docker tag "$IMAGE_TAG" "${IMAGE_NAME}:latest-${VARIANT}"

# Push images
echo "Pushing images to Docker Hub..."
docker push "$IMAGE_TAG"
docker push "${IMAGE_NAME}:latest-${VARIANT}"

echo "Successfully built and pushed:"
echo "  - ${IMAGE_TAG}"
echo "  - ${IMAGE_NAME}:latest-${VARIANT}"
