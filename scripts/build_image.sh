#!/bin/bash

ACR_NAME="monoaiinstance"
IMAGE_NAME="mono-ai-service"
TAG="latest"

# Build the image
docker build -t $IMAGE_NAME -f docker/Dockerfile .

# Tag the image
docker tag $IMAGE_NAME $ACR_NAME.azurecr.io/$IMAGE_NAME:$TAG