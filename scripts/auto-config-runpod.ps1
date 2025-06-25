# Script d'auto-configuration pour RunPod

# Paramètres
param(
    [string]$RunPodApiKey,
    [string]$RunPodEndpoint = "https://api.runpod.ai",
    [string]$ModelName = "mistral",
    [string]$AdminUsername = "admin",
    [string]$AdminPassword = "changeme",
    [string]$SecretKey = "your-secret-key-here"
)

# Vérification des prérequis
Write-Host "🔍 Vérification des prérequis..." -ForegroundColor Cyan

# Test de connexion à RunPod
Write-Host "📡 Test de connexion à RunPod..." -NoNewline
try {
    $response = Invoke-RestMethod -Uri "$RunPodEndpoint/health" -Method GET
    Write-Host "OK" -ForegroundColor Green
}
catch {
    Write-Host "ERREUR" -ForegroundColor Red
    Write-Host $_.Exception.Message
    exit 1
}

# Configuration des variables d'environnement
Write-Host "⚙️ Configuration des variables d'environnement..." -ForegroundColor Cyan

$envContent = @"
RUNPOD_API_KEY=$RunPodApiKey
RUNPOD_ENDPOINT=$RunPodEndpoint
MODEL_NAME=$ModelName
ADMIN_USERNAME=$AdminUsername
ADMIN_PASSWORD=$AdminPassword
SECRET_KEY=$SecretKey
"@

Set-Content -Path ".env" -Value $envContent

Write-Host "✅ Configuration terminée avec succès!" -ForegroundColor Green 