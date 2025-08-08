#!/bin/bash
set -e

echo "Starting Cloud Build..."
gcloud builds submit --config cloudbuild.yaml .
echo "Cloud Build finished successfully."
