Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "🔐 CONFIGURATION AZURE POUR GITHUB ACTIONS" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Vérifier si az CLI est installé
try {
    $azVersion = az version 2>$null
    Write-Host "✅ Azure CLI détecté" -ForegroundColor Green
} catch {
    Write-Host "❌ Azure CLI n'est pas installé. Veuillez l'installer depuis https://docs.microsoft.com/en-us/cli/azure/install-azure-cli" -ForegroundColor Red
    exit 1
}

# Vérifier si connecté à Azure
try {
    $account = az account show 2>$null | ConvertFrom-Json
    Write-Host "✅ Connecté à Azure: $($account.name)" -ForegroundColor Green
} catch {
    Write-Host "❌ Vous n'êtes pas connecté à Azure. Exécutez 'az login' d'abord." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Étape 1: Création du Service Principal..." -ForegroundColor Yellow

try {
    $spJson = az ad sp create-for-rbac `
        --name "bank-churn-github-sp-$(Get-Date -Format 'yyyyMMddHHmmss')" `
        --role contributor `
        --scopes "/subscriptions/10ed37ff-9840-4990-b7f6-1a238c5bdfa7" `
        --sdk-auth | ConvertFrom-Json

    Write-Host "✅ Service Principal créé avec succès!" -ForegroundColor Green
    Write-Host ""

    # Sauvegarder les credentials
    $spJson | ConvertTo-Json | Out-File -FilePath "azure_credentials.json" -Encoding UTF8

    Write-Host "📋 VALEURS À COPIER DANS GITHUB SECRETS:" -ForegroundColor Cyan
    Write-Host "=" * 50 -ForegroundColor White
    Write-Host ""

    Write-Host "AZURE_CREDENTIALS:" -ForegroundColor Yellow
    Write-Host ($spJson | ConvertTo-Json) -ForegroundColor White
    Write-Host ""

    Write-Host "=" * 50 -ForegroundColor White
    Write-Host ""
    Write-Host "🔑 OU COPIEZ INDIVIDUELLEMENT:" -ForegroundColor Cyan
    Write-Host ""

    Write-Host "AZURE_CLIENT_ID: $($spJson.clientId)" -ForegroundColor Green
    Write-Host "AZURE_CLIENT_SECRET: $($spJson.clientSecret)" -ForegroundColor Green
    Write-Host "AZURE_TENANT_ID: $($spJson.tenantId)" -ForegroundColor Green
    Write-Host "AZURE_SUBSCRIPTION_ID: $($spJson.subscriptionId)" -ForegroundColor Green

    Write-Host ""
    Write-Host "=" * 50 -ForegroundColor Cyan
    Write-Host "✅ TERMINÉ! Copiez ces valeurs dans GitHub" -ForegroundColor Green
    Write-Host "=" * 50 -ForegroundColor Cyan

} catch {
    Write-Host "❌ Erreur lors de la création du Service Principal: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "📁 Fichier azure_credentials.json créé avec vos credentials." -ForegroundColor Yellow
Write-Host ""
Write-Host "🚀 PROCHAINES ÉTAPES:" -ForegroundColor Green
Write-Host "1. Allez sur votre repo GitHub" -ForegroundColor White
Write-Host "2. Settings → Secrets and variables → Actions" -ForegroundColor White
Write-Host "3. Ajoutez les secrets listés ci-dessus" -ForegroundColor White
Write-Host "4. Poussez un commit pour déclencher le pipeline!" -ForegroundColor White