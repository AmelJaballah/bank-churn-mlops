"""
Module de détection de Data Drift
Utilise le test de Kolmogorov-Smirnov pour comparer les distributions
"""
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import json
import os
from datetime import datetime
from typing import Dict, Any

def detect_drift(
    reference_file: str,
    production_file: str,
    threshold: float = 0.05,
    output_dir: str = "drift_reports"
) -> Dict[str, Any]:
    """
    Détecte le drift entre données de référence et production
    
    Args:
        reference_file: Chemin vers le fichier CSV de référence
        production_file: Chemin vers le fichier CSV de production
        threshold: Seuil p-value pour détecter le drift (default: 0.05)
        output_dir: Dossier pour sauvegarder les rapports
    
    Returns:
        Dict avec les résultats de drift par feature
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Charger les données
    ref = pd.read_csv(reference_file)
    prod = pd.read_csv(production_file)
    
    results = {}
    features_with_drift = []
    
    # Colonnes numériques à analyser
    numeric_cols = ref.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        if col not in ["Exited", "Churn"] and col in prod.columns:
            # Test Kolmogorov-Smirnov
            stat, p_value = ks_2samp(
                ref[col].dropna(),
                prod[col].dropna()
            )
            
            drift_detected = p_value < threshold
            
            results[col] = {
                "ks_statistic": float(stat),
                "p_value": float(p_value),
                "drift_detected": drift_detected,
                "reference_mean": float(ref[col].mean()),
                "production_mean": float(prod[col].mean()),
                "reference_std": float(ref[col].std()),
                "production_std": float(prod[col].std())
            }
            
            if drift_detected:
                features_with_drift.append(col)
    
    # Résumé
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_features": len(results),
        "features_with_drift": len(features_with_drift),
        "drift_percentage": len(features_with_drift) / len(results) * 100 if results else 0,
        "drifted_features": features_with_drift,
        "threshold": threshold,
        "overall_status": "DRIFT_DETECTED" if features_with_drift else "NO_DRIFT"
    }
    
    # Sauvegarder le rapport
    report = {
        "summary": summary,
        "details": results
    }
    
    report_path = os.path.join(
        output_dir,
        f"drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"📊 Drift Report saved to: {report_path}")
    
    return results


def generate_drift_report(results: Dict[str, Any]) -> str:
    """Génère un rapport texte formaté"""
    
    features_with_drift = [f for f, r in results.items() if r.get("drift_detected")]
    
    report = f"""
╔══════════════════════════════════════════════════════════════╗
║                    DRIFT DETECTION REPORT                     ║
║                    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                       ║
╚══════════════════════════════════════════════════════════════╝

📊 SUMMARY
───────────────────────────────────────────────────────────────
  Total Features Analyzed: {len(results)}
  Features with Drift:     {len(features_with_drift)}
  Drift Percentage:        {len(features_with_drift)/len(results)*100:.1f}%
  Overall Status:          {"⚠️  DRIFT DETECTED" if features_with_drift else "✅ NO DRIFT"}

📋 DETAILED RESULTS
───────────────────────────────────────────────────────────────
"""
    
    for feature, data in results.items():
        status = "⚠️  DRIFT" if data["drift_detected"] else "✅ OK"
        report += f"""
  {feature}:
    Status:        {status}
    KS Statistic:  {data['ks_statistic']:.4f}
    P-Value:       {data['p_value']:.6f}
    Ref Mean:      {data['reference_mean']:.2f}
    Prod Mean:     {data['production_mean']:.2f}
"""
    
    if features_with_drift:
        report += f"""
⚠️  RECOMMENDATION
───────────────────────────────────────────────────────────────
  {len(features_with_drift)} feature(s) show significant drift.
  Consider retraining the model with recent data.
  Drifted features: {', '.join(features_with_drift)}
"""
    else:
        report += """
✅ RECOMMENDATION
───────────────────────────────────────────────────────────────
  No significant drift detected. Model is performing as expected.
  Continue regular monitoring.
"""
    
    return report