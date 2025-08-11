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


echo "Starting Cloud Build..."
gcloud builds submit --config cloudbuild.yaml \
  --substitutions=_PROJECT_ID=${PROJECT_ID},_LOCATION=${LOCATION},_REPOSITORY=${REPOSITORY},_IMAGE_NAME=${IMAGE_NAME} .
echo "Cloud Build finished successfully."
