"""
Drift Detection Module - Bank Churn MLOps
Détecte et simule le data drift pour le monitoring du modèle
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime
import os


class DriftDetector:
    """
    Classe pour détecter le Data Drift entre données de référence et données actuelles
    Utilise plusieurs tests statistiques: KS Test, Chi-Square, PSI
    """
    
    def __init__(self, reference_data: pd.DataFrame = None, reference_path: str = None):
        """
        Initialise le détecteur avec les données de référence
        
        Args:
            reference_data: DataFrame de référence
            reference_path: Chemin vers le fichier CSV de référence
        """
        if reference_data is not None:
            self.reference_data = reference_data
        elif reference_path:
            self.reference_data = pd.read_csv(reference_path)
        else:
            raise ValueError("Fournir reference_data ou reference_path")
        
        self.numeric_columns = self.reference_data.select_dtypes(include=[np.number]).columns.tolist()
        self.reference_stats = self._compute_statistics(self.reference_data)
        self.drift_threshold = 0.05  # p-value threshold
        self.psi_threshold = 0.2  # PSI threshold
        self.drift_history = []
    
    def _compute_statistics(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Calcule les statistiques descriptives pour chaque colonne numérique"""
        stats_dict = {}
        for col in self.numeric_columns:
            if col in data.columns:
                col_data = data[col].dropna()
                stats_dict[col] = {
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'median': float(col_data.median()),
                    'q25': float(col_data.quantile(0.25)),
                    'q75': float(col_data.quantile(0.75)),
                    'skew': float(col_data.skew()),
                    'kurtosis': float(col_data.kurtosis())
                }
        return stats_dict
    
    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """
        Calcule le Population Stability Index (PSI)
        PSI < 0.1: Pas de drift
        PSI 0.1-0.2: Drift modéré
        PSI > 0.2: Drift significatif
        """
        # Créer les bins basés sur les données de référence
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        # Calculer les proportions
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        curr_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Normaliser
        ref_props = ref_counts / len(reference)
        curr_props = curr_counts / len(current)
        
        # Éviter division par zéro
        ref_props = np.where(ref_props == 0, 0.0001, ref_props)
        curr_props = np.where(curr_props == 0, 0.0001, curr_props)
        
        # Calculer PSI
        psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
        
        return float(psi)
    
    def detect_drift_ks_test(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Détecte le drift en utilisant le test de Kolmogorov-Smirnov
        
        Returns:
            Dict avec les résultats du test pour chaque feature
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'test_type': 'Kolmogorov-Smirnov',
            'overall_drift': False,
            'features_with_drift': [],
            'feature_results': {}
        }
        
        for col in self.numeric_columns:
            if col in current_data.columns and col != 'Exited':
                ref_values = self.reference_data[col].dropna()
                curr_values = current_data[col].dropna()
                
                # Test KS
                ks_stat, p_value = stats.ks_2samp(ref_values, curr_values)
                
                drift_detected = p_value < self.drift_threshold
                
                results['feature_results'][col] = {
                    'ks_statistic': float(ks_stat),
                    'p_value': float(p_value),
                    'drift_detected': drift_detected,
                    'severity': 'HIGH' if p_value < 0.01 else ('MEDIUM' if p_value < 0.05 else 'LOW')
                }
                
                if drift_detected:
                    results['overall_drift'] = True
                    results['features_with_drift'].append(col)
        
        return results
    
    def detect_drift_psi(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Détecte le drift en utilisant le Population Stability Index (PSI)
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'test_type': 'Population Stability Index',
            'overall_drift': False,
            'features_with_drift': [],
            'feature_results': {}
        }
        
        for col in self.numeric_columns:
            if col in current_data.columns and col != 'Exited':
                ref_values = self.reference_data[col].dropna().values
                curr_values = current_data[col].dropna().values
                
                psi_value = self._calculate_psi(ref_values, curr_values)
                
                if psi_value < 0.1:
                    severity = 'LOW'
                    drift_detected = False
                elif psi_value < 0.2:
                    severity = 'MEDIUM'
                    drift_detected = True
                else:
                    severity = 'HIGH'
                    drift_detected = True
                
                results['feature_results'][col] = {
                    'psi_value': psi_value,
                    'drift_detected': drift_detected,
                    'severity': severity
                }
                
                if drift_detected:
                    results['overall_drift'] = True
                    results['features_with_drift'].append(col)
        
        return results
    
    def detect_drift_statistical(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Détecte le drift via comparaison des statistiques descriptives
        """
        current_stats = self._compute_statistics(current_data)
        results = {
            'timestamp': datetime.now().isoformat(),
            'test_type': 'Statistical Comparison',
            'overall_drift': False,
            'features_with_drift': [],
            'feature_results': {}
        }
        
        for col in self.numeric_columns:
            if col in current_stats and col != 'Exited':
                ref = self.reference_stats[col]
                curr = current_stats[col]
                
                # Calculer les changements
                mean_change = abs(curr['mean'] - ref['mean']) / (ref['std'] + 1e-6)
                std_ratio = curr['std'] / (ref['std'] + 1e-6)
                
                drift_detected = mean_change > 2.0 or std_ratio > 1.5 or std_ratio < 0.67
                
                results['feature_results'][col] = {
                    'mean_change_zscore': float(mean_change),
                    'std_ratio': float(std_ratio),
                    'ref_mean': ref['mean'],
                    'curr_mean': curr['mean'],
                    'ref_std': ref['std'],
                    'curr_std': curr['std'],
                    'drift_detected': drift_detected,
                    'severity': 'HIGH' if mean_change > 3 else ('MEDIUM' if mean_change > 2 else 'LOW')
                }
                
                if drift_detected:
                    results['overall_drift'] = True
                    results['features_with_drift'].append(col)
        
        return results
    
    def run_all_tests(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Exécute tous les tests de drift"""
        ks_results = self.detect_drift_ks_test(current_data)
        psi_results = self.detect_drift_psi(current_data)
        stat_results = self.detect_drift_statistical(current_data)
        
        combined = {
            'timestamp': datetime.now().isoformat(),
            'ks_test': ks_results,
            'psi_test': psi_results,
            'statistical_test': stat_results,
            'overall_drift': ks_results['overall_drift'] or psi_results['overall_drift'],
            'recommendation': 'RETRAIN MODEL' if (ks_results['overall_drift'] or psi_results['overall_drift']) else 'CONTINUE MONITORING'
        }
        
        # Sauvegarder dans l'historique
        self.drift_history.append(combined)
        
        return combined
    
    def generate_report(self, current_data: pd.DataFrame) -> str:
        """Génère un rapport textuel détaillé"""
        results = self.run_all_tests(current_data)
        
        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    DRIFT DETECTION REPORT                        ║
║                    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                          ║
╚══════════════════════════════════════════════════════════════════╝

📊 KOLMOGOROV-SMIRNOV TEST
{'─' * 66}
Overall Drift: {'⚠️ YES' if results['ks_test']['overall_drift'] else '✅ NO'}
Features with Drift: {', '.join(results['ks_test']['features_with_drift']) or 'None'}

"""
        for feat, res in results['ks_test']['feature_results'].items():
            status = '⚠️' if res['drift_detected'] else '✅'
            report += f"  {status} {feat}: KS={res['ks_statistic']:.4f}, p={res['p_value']:.6f} [{res['severity']}]\n"
        
        report += f"""
📈 POPULATION STABILITY INDEX (PSI)
{'─' * 66}
Overall Drift: {'⚠️ YES' if results['psi_test']['overall_drift'] else '✅ NO'}
Features with Drift: {', '.join(results['psi_test']['features_with_drift']) or 'None'}

"""
        for feat, res in results['psi_test']['feature_results'].items():
            status = '⚠️' if res['drift_detected'] else '✅'
            report += f"  {status} {feat}: PSI={res['psi_value']:.4f} [{res['severity']}]\n"
        
        report += f"""
📋 SUMMARY
{'─' * 66}
Overall Drift Detected: {'⚠️ YES' if results['overall_drift'] else '✅ NO'}
Recommendation: {results['recommendation']}

╚══════════════════════════════════════════════════════════════════╝
"""
        return report
    
    def save_results(self, filepath: str = 'drift_results.json'):
        """Sauvegarde les résultats dans un fichier JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.drift_history, f, indent=2, default=str)


def simulate_drift_scenarios(reference_path: str = 'data/bank_churn.csv'):
    """
    Simule différents scénarios de drift pour démonstration
    """
    print("=" * 70)
    print("🔬 DRIFT SIMULATION - Bank Churn MLOps")
    print("=" * 70)
    
    # Charger données de référence
    reference_data = pd.read_csv(reference_path)
    detector = DriftDetector(reference_data=reference_data)
    
    # Scénario 1: Pas de drift
    print("\n\n📊 SCÉNARIO 1: NO DRIFT (Données identiques)")
    print("-" * 70)
    no_drift_data = reference_data.copy()
    report1 = detector.generate_report(no_drift_data)
    print(report1)
    
    # Scénario 2: Drift léger
    print("\n\n📊 SCÉNARIO 2: SLIGHT DRIFT (Léger changement d'âge)")
    print("-" * 70)
    slight_drift = reference_data.copy()
    slight_drift['Age'] = slight_drift['Age'] + 3
    report2 = detector.generate_report(slight_drift)
    print(report2)
    
    # Scénario 3: Drift modéré
    print("\n\n📊 SCÉNARIO 3: MODERATE DRIFT (Changements multiples)")
    print("-" * 70)
    moderate_drift = reference_data.copy()
    moderate_drift['Age'] = moderate_drift['Age'] * 1.15
    moderate_drift['Balance'] = moderate_drift['Balance'] * 1.25
    report3 = detector.generate_report(moderate_drift)
    print(report3)
    
    # Scénario 4: Drift sévère
    print("\n\n📊 SCÉNARIO 4: SEVERE DRIFT (Changements majeurs)")
    print("-" * 70)
    severe_drift = reference_data.copy()
    severe_drift['Age'] = severe_drift['Age'] * 1.3
    severe_drift['CreditScore'] = severe_drift['CreditScore'] * 0.7
    severe_drift['Balance'] = severe_drift['Balance'] * 2
    severe_drift['EstimatedSalary'] = severe_drift['EstimatedSalary'] * 0.6
    report4 = detector.generate_report(severe_drift)
    print(report4)
    
    # Scénario 5: Distribution shift (nouvel échantillon aléatoire)
    print("\n\n📊 SCÉNARIO 5: DISTRIBUTION SHIFT (Nouvelle population)")
    print("-" * 70)
    np.random.seed(123)
    n = len(reference_data)
    distribution_shift = pd.DataFrame({
        'CreditScore': np.random.randint(400, 750, n),  # Distribution différente
        'Age': np.random.randint(25, 55, n),  # Plus jeune
        'Tenure': np.random.randint(0, 8, n),
        'Balance': np.random.uniform(10000, 100000, n),  # Moins varié
        'NumOfProducts': np.random.randint(1, 3, n),
        'HasCrCard': np.random.choice([0, 1], n, p=[0.3, 0.7]),
        'IsActiveMember': np.random.choice([0, 1], n, p=[0.4, 0.6]),
        'EstimatedSalary': np.random.uniform(30000, 100000, n),
        'Geography_Germany': np.random.choice([0, 1], n),
        'Geography_Spain': np.random.choice([0, 1], n),
    })
    report5 = detector.generate_report(distribution_shift)
    print(report5)
    
    # Sauvegarder les résultats
    detector.save_results('drift_simulation_results.json')
    
    print("\n" + "=" * 70)
    print(" Simulation terminée!")
    print(f" Résultats sauvegardés dans: drift_simulation_results.json")
    print("=" * 70)


if __name__ == "__main__":
    simulate_drift_scenarios()
