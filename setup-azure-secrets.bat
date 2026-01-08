@echo off
echo ========================================
echo AZURE MLOPS SETUP SCRIPT
echo ========================================
echo.

echo STEP 1: Creating Service Principal...
echo Replace YOUR_SUBSCRIPTION_ID with your actual subscription ID
echo.
echo Command to run:
echo az ad sp create-for-rbac --name "github-mlops-sp" --role contributor --scopes "/subscriptions/YOUR_SUBSCRIPTION_ID/resourceGroups/rg-mlops1" --sdk-auth
echo.
echo Copy the entire JSON output and use it as AZURE_CREDENTIALS secret in GitHub
echo.

echo STEP 2: Getting ACR Credentials...
echo.

set ACR_NAME=acrmlopsq1765467888

echo ACR Username:
az acr credential show --name %ACR_NAME% --query username -o tsv

echo.
echo ACR Password:
az acr credential show --name %ACR_NAME% --query "passwords[0].value" -o tsv

echo.
echo ========================================
echo GITHUB SECRETS TO CREATE:
echo ========================================
echo.
echo AZURE_CREDENTIALS = [JSON from step 1]
echo ACR_USERNAME = [username from step 2]
echo ACR_PASSWORD = [password from step 2]
echo.
echo ========================================
echo INSTRUCTIONS:
echo ========================================
echo.
echo 1. Run the az ad sp create-for-rbac command above
echo 2. Copy the JSON output
echo 3. Go to GitHub repo Settings -^> Secrets and variables -^> Actions
echo 4. Create the three secrets listed above
echo 5. Push any change to trigger the CI/CD pipeline
echo.
pause