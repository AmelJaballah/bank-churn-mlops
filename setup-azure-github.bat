@echo off
echo ===============================================
echo 🔐 CONFIGURATION AZURE POUR GITHUB ACTIONS
echo ===============================================
echo.

echo Étape 1: Création du Service Principal...
az ad sp create-for-rbac ^
  --name bank-churn-github-sp ^
  --role contributor ^
  --scopes /subscriptions/10ed37ff-9840-4990-b7f6-1a238c5bdfa7 ^
  --sdk-auth > azure_credentials.json

echo.
echo ✅ Service Principal créé!
echo.
echo 📋 COPIEZ CES VALEURS DANS GITHUB SECRETS:
echo ============================================
echo.

echo AZURE_CREDENTIALS:
type azure_credentials.json
echo.
echo ============================================
echo.
echo 🔑 OU COPIEZ INDIVIDUELLEMENT:
echo.
for /f "tokens=2 delims=:," %%a in ('findstr "clientId" azure_credentials.json') do (
    echo AZURE_CLIENT_ID: %%~a
)
for /f "tokens=2 delims=:," %%a in ('findstr "clientSecret" azure_credentials.json') do (
    echo AZURE_CLIENT_SECRET: %%~a
)
for /f "tokens=2 delims=:," %%a in ('findstr "tenantId" azure_credentials.json') do (
    echo AZURE_TENANT_ID: %%~a
)
for /f "tokens=2 delims=:," %%a in ('findstr "subscriptionId" azure_credentials.json') do (
    echo AZURE_SUBSCRIPTION_ID: %%~a
)

echo.
echo ============================================
echo ✅ TERMINÉ! Copiez ces valeurs dans GitHub
echo ============================================
pause