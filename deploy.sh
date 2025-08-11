#!/bin/bash
set -e

# Load environment variables from .env file if it exists
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Check for required environment variables
: "${PROJECT_ID?PROJECT_ID not set. Please set it in .env file or as an environment variable.}"
: "${LOCATION?LOCATION not set. Please set it in .env file or as an environment variable.}"
: "${REPOSITORY?REPOSITORY not set. Please set it in .env file or as an environment variable.}"
: "${IMAGE_NAME?IMAGE_NAME not set. Please set it in .env file or as an environment variable.}"
: "${SERVICE_NAME?SERVICE_NAME not set. Please set it in .env file or as an environment variable.}"

IMAGE_URL="${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:latest"

# Base gcloud command
GCLOUD_COMMAND="gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_URL} \
  --platform managed \
  --region ${LOCATION} \
  --memory 8Gi \
  --cpu 2 \
  --allow-unauthenticated"

# Add service account if it's set
if [ -n "${SERVICE_ACCOUNT_EMAIL}" ]; then
  echo "Using service account: ${SERVICE_ACCOUNT_EMAIL}"
  GCLOUD_COMMAND="${GCLOUD_COMMAND} --service-account ${SERVICE_ACCOUNT_EMAIL}"
else
  echo "No service account specified. Using default."
fi

echo "Deploying to Cloud Run with image: $IMAGE_URL"
# Execute the command
eval ${GCLOUD_COMMAND}

echo "Deployment to Cloud Run finished successfully."
