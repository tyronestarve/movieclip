#!/bin/bash
set -e

IMAGE_URL="us-central1-docker.pkg.dev/mylearningproject-429903/movieclip/movieclip:latest"

echo "Deploying to Cloud Run with image: $IMAGE_URL"
gcloud run deploy movieclip-service \
  --image $IMAGE_URL \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated

echo "Deployment to Cloud Run finished successfully."
