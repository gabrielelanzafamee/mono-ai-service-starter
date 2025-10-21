#!/bin/bash

ACR_NAME="monoaiinstance"
IMAGE_NAME="mono-ai-service"
TAG="latest"

# check first if the image exists
if docker image inspect $ACR_NAME.azurecr.io/$IMAGE_NAME:$TAG > /dev/null 2>&1; then
    echo "Image already exists"
else
    echo "Image does not exist"
    exit 1
fi

# Push the image to ACR
docker push $ACR_NAME.azurecr.io/$IMAGE_NAME:$TAG