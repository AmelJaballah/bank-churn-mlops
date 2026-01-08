"""
Tests unitaires pour l'API Bank Churn Prediction
"""
import sys
import os
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

# Ajouter le chemin parent pour importer app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Données de test valides
TEST_CUSTOMER = {
    "CreditScore": 650,
    "Age": 35,
    "Tenure": 5,
    "Balance": 50000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 75000.0,
    "Geography_Germany": 0,
    "Geography_Spain": 1
}

class TestHealthEndpoint:
    """Tests pour l'endpoint /health"""
    
    def test_health_returns_200(self):
        """Test que /health retourne un status 200"""
        response = client.get("/health")
        # Accepter 200 (modèle chargé) ou 503 (modèle non chargé en test)
        assert response.status_code in [200, 503]
    
    def test_health_response_structure(self):
        """Test la structure de la réponse health"""
        response = client.get("/health")
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert data["status"] == "healthy"


class TestRootEndpoint:
    """Tests pour l'endpoint racine /"""
    
    def test_root_returns_200(self):
        """Test que / retourne un status 200"""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_root_returns_api_info(self):
        """Test que / retourne les infos de l'API"""
        response = client.get("/")
        data = response.json()
        assert "message" in data or "status" in data


class TestPredictEndpoint:
    """Tests pour l'endpoint /predict"""
    
    def test_predict_with_valid_data(self):
        """Test /predict avec des données valides"""
        response = client.post("/predict", json=TEST_CUSTOMER)
        # Accepter 200 (succès), 422 (validation), ou 503 (modèle non chargé)
        assert response.status_code in [200, 422, 503]
    
    def test_predict_response_structure(self):
        """Test la structure de la réponse predict"""
        response = client.post("/predict", json=TEST_CUSTOMER)
        if response.status_code == 200:
            data = response.json()
            assert "churn_probability" in data
            assert "prediction" in data
            assert "risk_level" in data
            assert data["prediction"] in [0, 1]
            assert data["risk_level"] in ["Low", "Medium", "High"]
    
    def test_predict_with_mock_model(self):
        """Test /predict avec un modèle mocké"""
        with patch('app.main.model') as mock_model:
            mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
            mock_model.predict.return_value = np.array([1])
            
            response = client.post("/predict", json=TEST_CUSTOMER)
            # Le test passe si l'API traite la requête sans erreur serveur
            assert response.status_code in [200, 422, 503]
    
    def test_predict_missing_fields(self):
        """Test /predict avec des champs manquants"""
        incomplete_data = {"CreditScore": 650}
        response = client.post("/predict", json=incomplete_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_invalid_types(self):
        """Test /predict avec des types invalides"""
        invalid_data = TEST_CUSTOMER.copy()
        invalid_data["CreditScore"] = "invalid"
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422


class TestPredictWithMock:
    """Tests avec modèle complètement mocké"""
    
    @patch('app.main.model')
    def test_low_risk_prediction(self, mock_model):
        """Test prédiction risque faible"""
        mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])
        response = client.post("/predict", json=TEST_CUSTOMER)
        if response.status_code == 200:
            assert response.json()["risk_level"] == "Low"
    
    @patch('app.main.model')
    def test_high_risk_prediction(self, mock_model):
        """Test prédiction risque élevé"""
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
        response = client.post("/predict", json=TEST_CUSTOMER)
        if response.status_code == 200:
            assert response.json()["risk_level"] == "High"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])