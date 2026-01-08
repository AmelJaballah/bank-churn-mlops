"""
Bank Churn Prediction API
FastAPI application with ML model serving, monitoring and drift detection
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from typing import List
import logging
import os

from app.models import CustomerFeatures, PredictionResponse, HealthResponse

# ═══════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("bank-churn-api")

# Application Insights (optional)
APPINSIGHTS_CONN = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
if APPINSIGHTS_CONN:
    try:
        from opencensus.ext.azure.log_exporter import AzureLogHandler
        logger.addHandler(AzureLogHandler(connection_string=APPINSIGHTS_CONN))
        logger.info("✅ Application Insights connected")
    except ImportError:
        logger.warning("⚠️ opencensus not installed, Application Insights disabled")
else:
    logger.info("ℹ️ Application Insights not configured")

# ═══════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════
app = FastAPI(
    title="Bank Churn Prediction API",
    description="MLOps API for predicting customer churn in banking",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════
MODEL_PATH = os.getenv("MODEL_PATH", "model/model.pkl")
model = None

@app.on_event("startup")
async def load_model():
    """Load ML model on startup"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            logger.info(f"✅ Model loaded from {MODEL_PATH}")
        else:
            # Try alternative paths
            alt_paths = ["model/churn_model.pkl", "models/model.pkl"]
            for path in alt_paths:
                if os.path.exists(path):
                    model = joblib.load(path)
                    logger.info(f"✅ Model loaded from {path}")
                    break
            
            if model is None:
                logger.warning(f"⚠️ Model not found at {MODEL_PATH}")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        model = None

# ═══════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/", tags=["General"])
def root():
    """Root endpoint with API information"""
    return {
        "message": "Bank Churn Prediction API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded"
        )
    return {
        "status": "healthy",
        "model_loaded": True
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: CustomerFeatures):
    """
    Predict customer churn probability
    
    Returns:
        - churn_probability: Probability of churn (0-1)
        - prediction: Binary prediction (0=No Churn, 1=Churn)
        - risk_level: Low/Medium/High risk classification
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not available"
        )
    
    try:
        # Prepare features
        input_data = np.array([[
            features.CreditScore,
            features.Age,
            features.Tenure,
            features.Balance,
            features.NumOfProducts,
            features.HasCrCard,
            features.IsActiveMember,
            features.EstimatedSalary,
            features.Geography_Germany,
            features.Geography_Spain
        ]])
        
        # Prediction
        proba = model.predict_proba(input_data)[0, 1]
        prediction = int(proba > 0.5)
        
        # Risk classification
        if proba < 0.3:
            risk = "Low"
        elif proba < 0.7:
            risk = "Medium"
        else:
            risk = "High"
        
        logger.info(
            f"Prediction made: proba={proba:.4f}, "
            f"prediction={prediction}, risk={risk}"
        )
        
        return {
            "churn_probability": round(float(proba), 4),
            "prediction": prediction,
            "risk_level": risk
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(features_list: List[CustomerFeatures]):
    """
    Batch predictions for multiple customers
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        predictions = []
        
        for features in features_list:
            input_data = np.array([[
                features.CreditScore, features.Age, features.Tenure,
                features.Balance, features.NumOfProducts, features.HasCrCard,
                features.IsActiveMember, features.EstimatedSalary,
                features.Geography_Germany, features.Geography_Spain
            ]])
            
            proba = model.predict_proba(input_data)[0, 1]
            prediction = int(proba > 0.5)
            
            predictions.append({
                "churn_probability": round(float(proba), 4),
                "prediction": prediction
            })
        
        logger.info(f"Batch prediction: {len(predictions)} clients processed")
        
        return {"predictions": predictions, "count": len(predictions)}
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ═══════════════════════════════════════════════════════════════
# DRIFT DETECTION ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/drift/check", tags=["Monitoring"])
def check_drift(threshold: float = 0.05):
    """
    Check for data drift between reference and production data
    
    Args:
        threshold: P-value threshold for drift detection (default: 0.05)
    """
    try:
        from app.drift_detect import detect_drift
        
        reference_file = "data/bank_churn.csv"
        production_file = "data/production_data.csv"
        
        if not os.path.exists(production_file):
            return {
                "status": "no_data",
                "message": "No production data available for drift analysis"
            }
        
        results = detect_drift(
            reference_file=reference_file,
            production_file=production_file,
            threshold=threshold
        )
        
        drifted = [f for f, r in results.items() if r["drift_detected"]]
        drift_pct = len(drifted) / len(results) * 100 if results else 0
        
        logger.info(
            f"Drift check - Features analyzed: {len(results)}, "
            f"Drifted: {len(drifted)}, Percentage: {drift_pct:.1f}%"
        )
        
        return {
            "status": "success",
            "features_analyzed": len(results),
            "features_drifted": len(drifted),
            "drift_percentage": round(drift_pct, 2),
            "drifted_features": drifted,
            "risk_level": "HIGH" if drift_pct > 50 else "MEDIUM" if drift_pct > 20 else "LOW",
            "recommendation": "Retrain model" if drift_pct > 30 else "Continue monitoring"
        }
    
    except FileNotFoundError as e:
        return {"status": "error", "message": f"Data file not found: {e}"}
    except Exception as e:
        logger.error(f"Drift check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/drift/report", tags=["Monitoring"])
def get_drift_report():
    """Get the latest drift detection report"""
    import glob
    import json
    
    report_dir = "drift_reports"
    if not os.path.exists(report_dir):
        return {"status": "no_reports", "message": "No drift reports available"}
    
    reports = sorted(glob.glob(f"{report_dir}/drift_*.json"), reverse=True)
    
    if not reports:
        return {"status": "no_reports", "message": "No drift reports found"}
    
    with open(reports[0], "r") as f:
        latest_report = json.load(f)
    
    return {
        "status": "success",
        "report_file": reports[0],
        "report": latest_report
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)