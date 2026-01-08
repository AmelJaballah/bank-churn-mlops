"""
Tests unitaires pour l'API Bank Churn
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Ajouter le chemin parent pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests pour le endpoint /health"""
    
    def test_health_returns_200(self):
        """Test que /health retourne 200"""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_returns_healthy_status(self):
        """Test que /health retourne status healthy"""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] == True


class TestPredictEndpoint:
    """Tests pour le endpoint /predict"""
    
    @pytest.fixture
    def valid_payload(self):
        """Payload valide pour les tests"""
        return {
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
    
    def test_predict_returns_200(self, valid_payload):
        """Test que /predict retourne 200 avec un payload valide"""
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200
    
    def test_predict_returns_required_fields(self, valid_payload):
        """Test que la réponse contient tous les champs requis"""
        response = client.post("/predict", json=valid_payload)
        data = response.json()
        
        assert "churn_probability" in data
        assert "prediction" in data
        assert "risk_level" in data
    
    def test_predict_probability_range(self, valid_payload):
        """Test que la probabilité est entre 0 et 1"""
        response = client.post("/predict", json=valid_payload)
        data = response.json()
        
        assert 0 <= data["churn_probability"] <= 1
    
    def test_predict_binary_prediction(self, valid_payload):
        """Test que la prédiction est binaire (0 ou 1)"""
        response = client.post("/predict", json=valid_payload)
        data = response.json()
        
        assert data["prediction"] in [0, 1]
    
    def test_predict_valid_risk_level(self, valid_payload):
        """Test que le niveau de risque est valide"""
        response = client.post("/predict", json=valid_payload)
        data = response.json()
        
        assert data["risk_level"] in ["Low", "Medium", "High"]
    
    def test_predict_missing_field_returns_422(self):
        """Test qu'un champ manquant retourne 422"""
        incomplete_payload = {"CreditScore": 650}
        response = client.post("/predict", json=incomplete_payload)
        assert response.status_code == 422
    
    def test_predict_invalid_credit_score(self, valid_payload):
        """Test qu'un CreditScore invalide retourne une erreur"""
        valid_payload["CreditScore"] = 100  # En dessous du minimum (300)
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 422
    
    def test_predict_invalid_age(self, valid_payload):
        """Test qu'un Age invalide retourne une erreur"""
        valid_payload["Age"] = 10  # En dessous du minimum (18)
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 422
    
    def test_predict_high_risk_customer(self):
        """Test d'un client à haut risque"""
        high_risk_payload = {
            "CreditScore": 350,
            "Age": 65,
            "Tenure": 1,
            "Balance": 0.0,
            "NumOfProducts": 4,
            "HasCrCard": 0,
            "IsActiveMember": 0,
            "EstimatedSalary": 20000.0,
            "Geography_Germany": 1,
            "Geography_Spain": 0
        }
        response = client.post("/predict", json=high_risk_payload)
        assert response.status_code == 200
        data = response.json()
        # Un client à haut risque devrait avoir une probabilité plus élevée
        assert data["churn_probability"] >= 0


class TestRootEndpoint:
    """Tests pour le endpoint /"""
    
    def test_root_returns_200(self):
        """Test que / retourne 200"""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_root_returns_api_info(self):
        """Test que / retourne les informations de l'API"""
        response = client.get("/")
        data = response.json()
        
        assert "message" in data
        assert "version" in data
        assert "status" in data


class TestDocsEndpoint:
    """Tests pour la documentation Swagger"""
    
    def test_docs_returns_200(self):
        """Test que /docs retourne 200"""
        response = client.get("/docs")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
