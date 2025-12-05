#!/bin/bash
# TeleChat Docker Build and Publish Script
# Usage: ./build-and-publish.sh [cpu|gpu]
#
# This script builds and publishes Docker images to Docker Hub.
# Required environment variables:
#   DOCKERHUB_USERNAME - Your Docker Hub username
#   DOCKERHUB_TOKEN    - Your Docker Hub access token
#
# Optional environment variables:
#   IMAGE_NAME         - Docker image name (default: telechat)
#   IMAGE_TAG          - Docker image tag (default: latest)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print colored message
print_msg() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if variant argument is provided
if [ -z "$1" ]; then
    print_error "Usage: $0 [cpu|gpu]"
    print_msg "Example: $0 cpu"
    print_msg "Example: $0 gpu"
    exit 1
fi

VARIANT="$1"

# Validate variant
if [ "$VARIANT" != "cpu" ] && [ "$VARIANT" != "gpu" ]; then
    print_error "Invalid variant: $VARIANT"
    print_error "Must be either 'cpu' or 'gpu'"
    exit 1
fi

# Check required environment variables
if [ -z "$DOCKERHUB_USERNAME" ]; then
    print_error "DOCKERHUB_USERNAME environment variable is not set"
    print_msg "Set it with: export DOCKERHUB_USERNAME=your_username"
    exit 1
fi

if [ -z "$DOCKERHUB_TOKEN" ]; then
    print_error "DOCKERHUB_TOKEN environment variable is not set"
    print_msg "Set it with: export DOCKERHUB_TOKEN=your_token"
    print_msg "Generate a token at: https://hub.docker.com/settings/security"
    exit 1
fi

# Set default values
IMAGE_NAME="${IMAGE_NAME:-telechat}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Construct full image name with variant suffix
FULL_IMAGE_NAME="${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}-${VARIANT}"

print_msg "=========================================="
print_msg "TeleChat Docker Build and Publish"
print_msg "=========================================="
print_msg "Variant: $VARIANT"
print_msg "Image: $FULL_IMAGE_NAME"
print_msg "=========================================="

# Login to Docker Hub
print_msg "Logging into Docker Hub..."
# Note: Using stdin for password is recommended by Docker for security
# https://docs.docker.com/engine/reference/commandline/login/#provide-a-password-using-stdin
echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin

if [ $? -ne 0 ]; then
    print_error "Failed to login to Docker Hub"
    exit 1
fi

print_msg "✓ Successfully logged into Docker Hub"

# Determine Dockerfile
if [ "$VARIANT" = "gpu" ]; then
    DOCKERFILE="Dockerfile.full-gpu"
    print_warning "Building GPU image - this may take 30-60 minutes"
    print_warning "GPU required for building flash-attn and other CUDA dependencies"
else
    DOCKERFILE="Dockerfile.full-cpu"
    print_msg "Building CPU image - this may take 10-20 minutes"
fi

# Build the image
print_msg "Building Docker image from $DOCKERFILE..."
docker build -f "$DOCKERFILE" -t "$FULL_IMAGE_NAME" .

if [ $? -ne 0 ]; then
    print_error "Failed to build Docker image"
    exit 1
fi

print_msg "✓ Successfully built Docker image: $FULL_IMAGE_NAME"

# Push the image
print_msg "Pushing Docker image to Docker Hub..."
docker push "$FULL_IMAGE_NAME"

if [ $? -ne 0 ]; then
    print_error "Failed to push Docker image"
    exit 1
fi

print_msg "✓ Successfully pushed Docker image: $FULL_IMAGE_NAME"

# Tag as latest-variant if IMAGE_TAG is 'latest'
if [ "$IMAGE_TAG" = "latest" ]; then
    LATEST_IMAGE="${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest-${VARIANT}"
    print_msg "Tagging as $LATEST_IMAGE..."
    docker tag "$FULL_IMAGE_NAME" "$LATEST_IMAGE"
    docker push "$LATEST_IMAGE"
    print_msg "✓ Also pushed as: $LATEST_IMAGE"
fi

print_msg "=========================================="
print_msg "Build and publish completed successfully!"
print_msg "=========================================="
print_msg "Image: $FULL_IMAGE_NAME"
print_msg "Pull command: docker pull $FULL_IMAGE_NAME"
print_msg "Run command: docker run -p 8000:8000 $FULL_IMAGE_NAME"
print_msg "=========================================="
