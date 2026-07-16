#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# IMPORTANT: Set these variables before running the script.

# Your Google Cloud project ID.
export PROJECT_ID=$(gcloud config get-value project)

# The Google Cloud region for your resources.
export REGION="us-central1"

# The name for your GCS bucket (must be globally unique).
export GCS_BUCKET_NAME="your-gcs-bucket-name-here"

# The name for your service, repository, and service account.
export SERVICE_NAME="ai-act-processor"

# --- Derived Variables ---
export REPO_NAME="${SERVICE_NAME}-repo"
export IMAGE_NAME="${SERVICE_NAME}"
export SERVICE_ACCOUNT_EMAIL="${SERVICE_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# --- Script ---

echo "Enabling required Google Cloud services..."
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com iam.googleapis.com

echo "Creating Artifact Registry repository..."
gcloud artifacts repositories create "${REPO_NAME}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="Docker repository for the AI Act Processor application" || echo "Repository '${REPO_NAME}' already exists."

echo "Creating service account..."
gcloud iam service-accounts create ${SERVICE_NAME} \
    --display-name="AI Act Processor Service Account" || echo "Service account '${SERVICE_NAME}' already exists."

echo "Creating GCS bucket if it doesn't exist..."
gcloud storage buckets create "gs://${GCS_BUCKET_NAME}" --project="${PROJECT_ID}" --location="${REGION}" --uniform-bucket-level-access || echo "Bucket 'gs://${GCS_BUCKET_NAME}' already exists or you lack permissions."

echo "Granting GCS bucket permissions..."
gcloud storage buckets add-iam-policy-binding "gs://${GCS_BUCKET_NAME}" \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/storage.objectCreator"

echo "Granting Vertex AI permissions..."
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/aiplatform.user"

echo "Building and submitting container image..."
gcloud builds submit --tag "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest"

echo "Deploying Cloud Run job..."
gcloud run jobs deploy "${SERVICE_NAME}" \
    --image "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest" \
    --region "${REGION}" \
    --service-account "${SERVICE_ACCOUNT_EMAIL}" \
    --set-env-vars="GCS_BUCKET=${GCS_BUCKET_NAME},GCP_PROJECT=${PROJECT_ID},GCP_LOCATION=${REGION}"

echo "Deployment complete. To run the job, use the following command:"
echo "gcloud run jobs execute \"${SERVICE_NAME}\" --region \"${REGION}\""