#!/bin/bash
# TeleChat Docker Build and Publish Script
# =========================================
# Usage: ./build-and-publish.sh [cpu|gpu] [tag]
#
# This script builds and publishes Docker images to Docker Hub
# Requires DOCKERHUB_USERNAME and DOCKERHUB_TOKEN environment variables

set -e

# Configuration
VARIANT="${1:-cpu}"
TAG="${2:-latest}"
IMAGE_NAME="${DOCKERHUB_USERNAME}/telechat"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
error() {
    echo -e "${RED}❌ ERROR: $1${NC}" >&2
    exit 1
}

info() {
    echo -e "${GREEN}✅ $1${NC}"
}

warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Validation
if [[ "$VARIANT" != "cpu" && "$VARIANT" != "gpu" ]]; then
    error "Invalid variant '$VARIANT'. Must be 'cpu' or 'gpu'"
fi

# Check for required environment variables
if [[ -z "$DOCKERHUB_USERNAME" ]]; then
    error "DOCKERHUB_USERNAME environment variable is not set"
fi

if [[ -z "$DOCKERHUB_TOKEN" ]]; then
    error "DOCKERHUB_TOKEN environment variable is not set"
fi

# Determine Dockerfile
if [[ "$VARIANT" == "gpu" ]]; then
    DOCKERFILE="Dockerfile.full-gpu"
    IMAGE_TAG="${IMAGE_NAME}:${TAG}-gpu"
else
    DOCKERFILE="Dockerfile.full-cpu"
    IMAGE_TAG="${IMAGE_NAME}:${TAG}-cpu"
fi

# Check if Dockerfile exists
if [[ ! -f "$DOCKERFILE" ]]; then
    error "Dockerfile not found: $DOCKERFILE"
fi

info "Starting Docker build and publish process"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Variant:    $VARIANT"
echo "Dockerfile: $DOCKERFILE"
echo "Image tag:  $IMAGE_TAG"
echo "Registry:   Docker Hub"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Login to Docker Hub
info "Logging in to Docker Hub..."
echo "$DOCKERHUB_TOKEN" | docker login --username "$DOCKERHUB_USERNAME" --password-stdin || error "Docker Hub login failed"

# Build image
info "Building Docker image..."
if [[ "$VARIANT" == "gpu" ]]; then
    warn "GPU build may take 30-60+ minutes due to CUDA compilation"
fi

docker build -f "$DOCKERFILE" -t "$IMAGE_TAG" . || error "Docker build failed"

info "Build completed successfully"

# Tag with additional tags if specified
if [[ "$TAG" != "latest" ]]; then
    LATEST_TAG="${IMAGE_NAME}:latest-${VARIANT}"
    info "Tagging as $LATEST_TAG"
    docker tag "$IMAGE_TAG" "$LATEST_TAG"
fi

# Push to Docker Hub
info "Pushing image to Docker Hub..."
docker push "$IMAGE_TAG" || error "Docker push failed"

if [[ "$TAG" != "latest" ]]; then
    docker push "$LATEST_TAG" || error "Docker push failed"
fi

info "Image published successfully!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Published: $IMAGE_TAG"
if [[ "$TAG" != "latest" ]]; then
    echo "Also tagged: $LATEST_TAG"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Logout
docker logout || warn "Docker logout failed (non-critical)"

info "Done!"
