"""
Drift Detection Module for Bank Churn API
Provides functions to detect data drift between reference and production data
"""
import pandas as pd
import numpy as np
from scipy import stats
import json
import os
from datetime import datetime
from typing import Dict, List, Optional


# Numerical features to analyze for drift
NUMERICAL_FEATURES = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 
    'NumOfProducts', 'EstimatedSalary'
]


def calculate_psi(reference: np.array, production: np.array, buckets: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI)
    
    PSI < 0.1: No significant shift
    PSI 0.1-0.25: Moderate shift
    PSI > 0.25: Significant shift
    """
    # Define bucket boundaries from reference data
    min_val = min(reference.min(), production.min())
    max_val = max(reference.max(), production.max())
    
    breakpoints = np.linspace(min_val, max_val, buckets + 1)
    
    # Calculate percentages in each bucket
    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    prod_counts = np.histogram(production, bins=breakpoints)[0]
    
    # Convert to percentages (avoid division by zero)
    ref_pct = (ref_counts + 0.0001) / len(reference)
    prod_pct = (prod_counts + 0.0001) / len(production)
    
    # Calculate PSI
    psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    
    return float(psi)


def detect_drift(
    reference_file: str,
    production_file: str,
    threshold: float = 0.05,
    features: Optional[List[str]] = None
) -> Dict:
    """
    Detect drift between reference and production data using KS test
    
    Args:
        reference_file: Path to reference dataset (CSV)
        production_file: Path to production dataset (CSV)
        threshold: P-value threshold for drift detection
        features: List of features to analyze (default: NUMERICAL_FEATURES)
    
    Returns:
        Dictionary with drift results for each feature
    """
    # Load data
    ref_data = pd.read_csv(reference_file)
    prod_data = pd.read_csv(production_file)
    
    if features is None:
        features = [f for f in NUMERICAL_FEATURES if f in ref_data.columns and f in prod_data.columns]
    
    results = {}
    
    for feature in features:
        if feature in ref_data.columns and feature in prod_data.columns:
            ref_values = ref_data[feature].dropna().values
            prod_values = prod_data[feature].dropna().values
            
            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(ref_values, prod_values)
            
            # PSI calculation
            psi = calculate_psi(ref_values, prod_values)
            
            # Determine drift
            drift_detected = p_value < threshold or psi > 0.25
            
            results[feature] = {
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": round(float(p_value), 6),
                "psi": round(float(psi), 4),
                "drift_detected": drift_detected,
                "severity": "HIGH" if psi > 0.25 else "MEDIUM" if psi > 0.1 else "LOW"
            }
    
    return results


def generate_drift_report(
    reference_file: str,
    production_file: str,
    threshold: float = 0.05,
    output_dir: str = "drift_reports"
) -> str:
    """
    Generate a comprehensive drift report
    
    Returns:
        Path to the generated report file
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Detect drift
    results = detect_drift(reference_file, production_file, threshold)
    
    # Compile report
    drifted_features = [f for f, r in results.items() if r["drift_detected"]]
    drift_percentage = len(drifted_features) / len(results) * 100 if results else 0
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "reference_file": reference_file,
        "production_file": production_file,
        "threshold": threshold,
        "summary": {
            "features_analyzed": len(results),
            "features_drifted": len(drifted_features),
            "drift_percentage": round(drift_percentage, 2),
            "drifted_features": drifted_features,
            "overall_status": "DRIFT_DETECTED" if drifted_features else "STABLE",
            "recommendation": "Retrain model recommended" if drift_percentage > 30 else "Continue monitoring"
        },
        "details": results
    }
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"drift_report_{timestamp}.json")
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    return report_path


def generate_production_data_with_drift(
    reference_file: str,
    output_file: str,
    drift_intensity: float = 0.3,
    n_samples: int = 500
) -> str:
    """
    Generate synthetic production data with controlled drift
    
    Args:
        reference_file: Path to reference dataset
        output_file: Output path for generated data
        drift_intensity: How much drift to introduce (0-1)
        n_samples: Number of samples to generate
    
    Returns:
        Path to generated file
    """
    ref_data = pd.read_csv(reference_file)
    
    # Sample from reference data
    prod_data = ref_data.sample(n=min(n_samples, len(ref_data)), replace=True).copy()
    
    # Introduce drift in numerical features
    for feature in NUMERICAL_FEATURES:
        if feature in prod_data.columns:
            # Shift mean and increase variance
            mean_shift = prod_data[feature].std() * drift_intensity * np.random.choice([-1, 1])
            prod_data[feature] = prod_data[feature] + mean_shift
            
            # Add some noise
            noise = np.random.normal(0, prod_data[feature].std() * drift_intensity * 0.5, len(prod_data))
            prod_data[feature] = prod_data[feature] + noise
    
    # Save
    prod_data.to_csv(output_file, index=False)
    
    return output_file
