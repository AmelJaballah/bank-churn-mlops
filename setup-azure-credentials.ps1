# PowerShell script to setup Azure credentials for GitHub Actions
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "AZURE MLOPS CREDENTIALS SETUP" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Get subscription ID
Write-Host "Getting your Azure subscription ID..." -ForegroundColor Yellow
$subscriptionId = az account show --query id -o tsv
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Error: Please login to Azure first with 'az login'" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Subscription ID: $subscriptionId" -ForegroundColor Green
Write-Host ""

# Create Service Principal
Write-Host "Creating Service Principal with Contributor role..." -ForegroundColor Yellow
$spJson = az ad sp create-for-rbac --name "github-mlops-sp-$(Get-Date -Format 'yyyyMMddHHmmss')" --role contributor --scopes "/subscriptions/$subscriptionId/resourceGroups/rg-mlops1" --sdk-auth
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Error creating Service Principal" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Service Principal created!" -ForegroundColor Green
Write-Host ""

# Get ACR credentials
Write-Host "Getting ACR credentials..." -ForegroundColor Yellow
$acrName = "acrmlopsq1765467888"
$acrUsername = az acr credential show --name $acrName --query username -o tsv
$acrPassword = az acr credential show --name $acrName --query "passwords[0].value" -o tsv
Write-Host "✅ ACR credentials retrieved!" -ForegroundColor Green
Write-Host ""

# Display results
Write-Host "=========================================" -ForegroundColor Green
Write-Host "GITHUB SECRETS TO CREATE:" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
Write-Host "1. AZURE_CREDENTIALS:" -ForegroundColor Cyan
Write-Host "$spJson" -ForegroundColor White
Write-Host ""
Write-Host "2. ACR_USERNAME:" -ForegroundColor Cyan
Write-Host "$acrUsername" -ForegroundColor White
Write-Host ""
Write-Host "3. ACR_PASSWORD:" -ForegroundColor Cyan
Write-Host "$acrPassword" -ForegroundColor White
Write-Host ""

Write-Host "=========================================" -ForegroundColor Yellow
Write-Host "INSTRUCTIONS:" -ForegroundColor Yellow
Write-Host "=========================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Copy the AZURE_CREDENTIALS JSON above" -ForegroundColor White
Write-Host "2. Go to GitHub → Your Repo → Settings → Secrets and variables → Actions" -ForegroundColor White
Write-Host "3. Create the three secrets listed above" -ForegroundColor White
Write-Host "4. Push any change to trigger the CI/CD pipeline" -ForegroundColor White
Write-Host ""
Write-Host "✅ Setup complete!" -ForegroundColor Green