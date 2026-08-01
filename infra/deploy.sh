#!/bin/bash
# One-time Azure infrastructure setup script
# Prerequisites: az CLI installed, az login completed

set -e

RESOURCE_GROUP="docopilot-rg"
LOCATION="centralindia"
ACR_NAME="docopilotacr"
APP_ENV="docopilot-env"
APP_NAME="docopilot-backend"


echo "Creating Resource Group..."
az group create --name $RESOURCE_GROUP --location $LOCATION

echo "Creating Azure Container Registry..."
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --sku Basic \
  --admin-enabled true

echo "Creating Container Apps Environment..."
az containerapp env create \
  --name $APP_ENV \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION

echo "Creating Container App (initial deploy with placeholder)..."
az containerapp create \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --environment $APP_ENV \
  --image mcr.microsoft.com/azuredocs/containerapps-helloworld:latest \
  --target-port 8000 \
  --ingress external \
  --min-replicas 0 \
  --max-replicas 3 \
  --cpu 0.5 \
  --memory 1.0Gi \
  --env-vars \
    QDRANT_URL=secretref:qdrant-url \
    DATABASE_URL=secretref:database-url \
    JWT_SECRET_KEY=secretref:jwt-secret \
    GROQ_API_KEY=secretref:groq-api-key \
    COHERE_API_KEY=secretref:cohere-api-key \
    ALLOWED_ORIGINS=https://your-frontend.vercel.app

echo "Done. Update secrets with: az containerapp secret set ..."
echo "Then push first image via: az acr build --registry $ACR_NAME ..."
