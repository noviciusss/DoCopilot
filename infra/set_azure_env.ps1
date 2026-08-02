# PowerShell script to update Azure Container App environment variables
# Uses environment variables if set, otherwise prompts or uses default placeholders

param(
  [string]$DatabaseUrl = $env:DATABASE_URL,
  [string]$GroqApiKey  = $env:GROQ_API_KEY,
  [string]$CohereApiKey = $env:COHERE_API_KEY,
  [string]$JwtSecret   = $env:JWT_SECRET_KEY
)

$AZ_RESOURCE_GROUP = "docopilot-rg"
$AZ_APP_NAME       = "docopilot-backend"
$AZ_ACR_REGISTRY   = "docopilotacr.azurecr.io"
$AZ_ACR_USER       = "docopilotacr"
$AZ_ACR_PASS       = "53MbjD3A2fkS0vvdiVsY3jBT07KEJU3VKuAQe3vYkCtVUvT4QVvlJQQJ99CHACGhslBEqg7NAAACAZCRTMPo"
$IMAGE_NAME        = "$AZ_ACR_REGISTRY/docopilot-backend:latest"

Write-Host "Updating Azure Container App environment variables..."
az containerapp update `
  --name $AZ_APP_NAME `
  --resource-group $AZ_RESOURCE_GROUP `
  --set-env-vars `
    "DATABASE_URL=$DatabaseUrl" `
    "GROQ_API_KEY=$GroqApiKey" `
    "COHERE_API_KEY=$CohereApiKey" `
    "JWT_SECRET_KEY=$JwtSecret" `
    "JWT_ALGORITHM=HS256" `
    "ACCESS_TOKEN_EXPIRE_MINUTES=1440" `
    "ALLOWED_ORIGINS=https://do-copilot.vercel.app,http://localhost:3000" `
    "LOG_LEVEL=INFO" `
    "MAX_UPLOAD_SIZE_MB=20"
