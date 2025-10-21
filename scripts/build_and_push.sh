#!/bin/bash

# Build the image
./scripts/build_image.sh

# Push the image to ACR
./scripts/push_image_to_acr.sh