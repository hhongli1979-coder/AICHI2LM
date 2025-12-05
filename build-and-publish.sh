#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-cpu}"
IMAGE="${IMAGE:-hhongli1979-coder/aichi2lm}"
TAG="${TAG:-latest}"

DOCKERHUB_USER="${DOCKERHUB_USERNAME:-}"
DOCKERHUB_TOKEN="${DOCKERHUB_TOKEN:-}"

if [ -z "$DOCKERHUB_USER" ] || [ -z "$DOCKERHUB_TOKEN" ]; then
  echo "Please set DOCKERHUB_USERNAME and DOCKERHUB_TOKEN environment variables."
  exit 1
fi

echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USER" --password-stdin

if [ "$MODE" = "gpu" ]; then
  IMAGE_TAG="${IMAGE}:${TAG}-gpu"
  echo "Building GPU image ${IMAGE_TAG} ..."
  docker build -f Dockerfile.full-gpu -t "${IMAGE_TAG}" .
  docker push "${IMAGE_TAG}"
else
  IMAGE_TAG="${IMAGE}:${TAG}-cpu"
  echo "Building CPU image ${IMAGE_TAG} ..."
  docker build -f Dockerfile.full-cpu -t "${IMAGE_TAG}" .
  docker push "${IMAGE_TAG}"
fi

echo "Done: pushed ${IMAGE_TAG}"
