"""
Bank Churn Prediction - Streamlit Interface
Application web pour la prédiction de défaillance client (churn)
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any
import os
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Bank Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL de l'API (variable d'environnement ou valeur par défaut)
API_URL = os.getenv("API_URL", "https://bank-churn.salmonbay-9a939e1f.francecentral.azurecontainerapps.io")

# CSS personnalisé amélioré
st.markdown("""
<style>
    /* Global Styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header Styling */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ffffff, #e0e7ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        color: rgba(255,255,255,0.8);
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Card Styling */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f2937;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6b7280;
        margin-top: 5px;
    }
    
    /* Risk Badges */
    .risk-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1rem;
    }
    
    .risk-low { background: linear-gradient(135deg, #10b981, #059669); color: white; }
    .risk-medium { background: linear-gradient(135deg, #f59e0b, #d97706); color: white; }
    .risk-high { background: linear-gradient(135deg, #ef4444, #dc2626); color: white; }
    
    /* Drift Status */
    .drift-ok { background: linear-gradient(135deg, #10b981, #059669); color: white; padding: 10px 20px; border-radius: 10px; }
    .drift-warning { background: linear-gradient(135deg, #f59e0b, #d97706); color: white; padding: 10px 20px; border-radius: 10px; }
    .drift-critical { background: linear-gradient(135deg, #ef4444, #dc2626); color: white; padding: 10px 20px; border-radius: 10px; }
    
    /* Button Styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e2e 0%, #2d2d44 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: white;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: rgba(255,255,255,0.7);
        padding: 20px;
        margin-top: 2rem;
    }
    
    /* Animations */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse { animation: pulse 2s infinite; }
</style>
""", unsafe_allow_html=True)

def check_api_health() -> bool:
    """Vérifie si l'API est disponible"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_api_info() -> Dict[str, Any]:
    """Récupère les infos de l'API"""
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        return response.json()
    except:
        return None

def make_prediction(features: Dict[str, Any]) -> Dict[str, Any]:
    """Envoie une requête de prédiction à l'API"""
    try:
        # Convert all values to int/float as needed
        payload = {
            "CreditScore": int(features["CreditScore"]),
            "Age": int(features["Age"]),
            "Tenure": int(features["Tenure"]),
            "Balance": float(features["Balance"]),
            "NumOfProducts": int(features["NumOfProducts"]),
            "HasCrCard": int(features["HasCrCard"]),
            "IsActiveMember": int(features["IsActiveMember"]),
            "EstimatedSalary": float(features["EstimatedSalary"]),
            "Geography_Germany": int(features["Geography_Germany"]),
            "Geography_Spain": int(features["Geography_Spain"])
        }
        
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API: {e}")
        return None

def check_drift() -> Dict[str, Any]:
    """Vérifie le drift des données via l'API"""
    try:
        response = requests.post(f"{API_URL}/drift/check", timeout=30)
        return response.json()
    except:
        return None

def generate_synthetic_drift_data(reference_data: pd.DataFrame, drift_intensity: float = 0.3) -> pd.DataFrame:
    """Génère des données synthétiques avec drift"""
    n_samples = min(500, len(reference_data))
    prod_data = reference_data.sample(n=n_samples, replace=True).copy()
    
    numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    
    for feature in numerical_features:
        if feature in prod_data.columns:
            mean_shift = prod_data[feature].std() * drift_intensity * np.random.choice([-1, 1])
            prod_data[feature] = prod_data[feature] + mean_shift
            noise = np.random.normal(0, prod_data[feature].std() * drift_intensity * 0.3, len(prod_data))
            prod_data[feature] = prod_data[feature] + noise
    
    return prod_data

def calculate_psi(reference: np.array, production: np.array, buckets: int = 10) -> float:
    """Calculate Population Stability Index"""
    min_val = min(reference.min(), production.min())
    max_val = max(reference.max(), production.max())
    breakpoints = np.linspace(min_val, max_val, buckets + 1)
    
    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    prod_counts = np.histogram(production, bins=breakpoints)[0]
    
    ref_pct = (ref_counts + 0.0001) / len(reference)
    prod_pct = (prod_counts + 0.0001) / len(production)
    
    psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    return float(psi)

def create_gauge_chart(probability: float) -> go.Figure:
    """Crée un graphique jauge pour la probabilité de churn"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': '%', 'font': {'size': 40, 'color': '#1f2937'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#374151", 'tickfont': {'size': 14}},
            'bar': {'color': "#667eea", 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': "#e5e7eb",
            'steps': [
                {'range': [0, 30], 'color': '#d1fae5'},
                {'range': [30, 60], 'color': '#fef3c7'},
                {'range': [60, 100], 'color': '#fee2e2'}
            ],
            'threshold': {
                'line': {'color': "#ef4444", 'width': 4},
                'thickness': 0.8,
                'value': probability * 100
            }
        }
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#1f2937'}
    )
    return fig

def create_drift_comparison_chart(ref_data: pd.DataFrame, prod_data: pd.DataFrame, feature: str) -> go.Figure:
    """Crée un graphique de comparaison entre données de référence et production"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=ref_data[feature],
        name='Référence',
        opacity=0.7,
        marker_color='#667eea',
        nbinsx=30
    ))
    
    fig.add_trace(go.Histogram(
        x=prod_data[feature],
        name='Production',
        opacity=0.7,
        marker_color='#ef4444',
        nbinsx=30
    ))
    
    fig.update_layout(
        title=f'Distribution: {feature}',
        barmode='overlay',
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title=feature,
        yaxis_title='Fréquence',
        legend=dict(x=0.7, y=0.95),
        margin=dict(l=50, r=20, t=50, b=50)
    )
    
    return fig

def create_drift_summary_chart(drift_results: Dict) -> go.Figure:
    """Crée un graphique résumé du drift"""
    features = list(drift_results.keys())
    psi_values = [drift_results[f]['psi'] for f in features]
    colors = ['#ef4444' if v > 0.25 else '#f59e0b' if v > 0.1 else '#10b981' for v in psi_values]
    
    fig = go.Figure(go.Bar(
        x=psi_values,
        y=features,
        orientation='h',
        marker_color=colors,
        text=[f'{v:.3f}' for v in psi_values],
        textposition='outside'
    ))
    
    # Add threshold lines
    fig.add_vline(x=0.1, line_dash="dash", line_color="#f59e0b", annotation_text="Warning", annotation_position="top")
    fig.add_vline(x=0.25, line_dash="dash", line_color="#ef4444", annotation_text="Critical", annotation_position="top")
    
    fig.update_layout(
        title='PSI par Feature (Population Stability Index)',
        xaxis_title='PSI',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.9)',
        margin=dict(l=120, r=50, t=50, b=50)
    )
    
    return fig

def create_feature_importance_chart(features: Dict[str, Any]) -> go.Figure:
    """Crée un graphique radar du profil client"""
    # Normalisation pour radar chart
    normalized = {
        'Credit Score': min(features['CreditScore'] / 850 * 100, 100),
        'Âge': min(features['Age'] / 80 * 100, 100),
        'Ancienneté': min(features['Tenure'] / 10 * 100, 100),
        'Solde': min(features['Balance'] / 200000 * 100, 100),
        'Produits': min(features['NumOfProducts'] / 4 * 100, 100),
        'Engagement': features['IsActiveMember'] * 100,
    }
    
    categories = list(normalized.keys())
    values = list(normalized.values())
    values.append(values[0])  # Close the radar
    categories.append(categories[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color='#667eea', width=2),
        name='Profil Client'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10)
            ),
            angularaxis=dict(tickfont=dict(size=11))
        ),
        showlegend=False,
        height=320,
        margin=dict(l=60, r=60, t=30, b=30),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig
def main():
    # Header
    st.markdown('<h1 class="main-header">🏦 Bank Churn Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Prédiction intelligente de défaillance client avec détection de drift</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🔗 État du Système")
        api_status = check_api_health()
        api_info = get_api_info()
        
        if api_status:
            st.success("✅ API Connectée")
            if api_info:
                st.info(f"📌 Version: {api_info.get('version', 'N/A')}")
        else:
            st.error("❌ API Non Disponible")
        
        st.markdown("---")
        
        st.markdown("## 📊 Légende des Risques")
        st.markdown("""
        <div style='padding: 10px; background: rgba(255,255,255,0.1); border-radius: 10px;'>
            <p>🟢 <strong>Low</strong>: < 30% de risque</p>
            <p>🟡 <strong>Medium</strong>: 30-60% de risque</p>
            <p>🔴 <strong>High</strong>: > 60% de risque</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("## ℹ️ À propos")
        st.markdown("""
        <div style='padding: 10px; background: rgba(255,255,255,0.1); border-radius: 10px; font-size: 0.9rem;'>
            <p><strong>Modèle:</strong> Random Forest</p>
            <p><strong>Framework:</strong> FastAPI + MLflow</p>
            <p><strong>Cloud:</strong> Azure Container Apps</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Prédiction", "📊 Analyse Drift", "📁 Batch"])
    
    # ═══════════════════════════════════════════════════════════════
    # TAB 1: PREDICTION
    # ═══════════════════════════════════════════════════════════════
    with tab1:
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("### 📋 Informations Client")
            
            # Informations personnelles
            st.markdown("#### 👤 Profil Personnel")
            age = st.slider("Âge", min_value=18, max_value=100, value=35, help="Âge du client", key="age_tab1")
            
            col_geo1, col_geo2 = st.columns(2)
            with col_geo1:
                geography = st.selectbox(
                    "Pays",
                    options=["France", "Germany", "Spain"],
                    help="Pays de résidence du client"
                )
            with col_geo2:
                tenure = st.slider("Ancienneté (années)", min_value=0, max_value=10, value=5, key="tenure_tab1")
            
            # Informations financières
            st.markdown("#### 💰 Informations Financières")
            credit_score = st.slider(
                "Score de Crédit",
                min_value=300, max_value=850, value=650,
                help="Score de crédit (300-850)",
                key="credit_tab1"
            )
            
            col_fin1, col_fin2 = st.columns(2)
            with col_fin1:
                balance = st.number_input(
                    "Solde du compte (€)",
                    min_value=0.0, max_value=500000.0, value=50000.0,
                    step=1000.0,
                    help="Solde actuel du compte"
                )
            with col_fin2:
                estimated_salary = st.number_input(
                    "Salaire estimé (€)",
                    min_value=0.0, max_value=300000.0, value=75000.0,
                    step=1000.0,
                    help="Salaire annuel estimé"
                )
            
            # Produits et services
            st.markdown("#### 🏦 Produits & Services")
            col_prod1, col_prod2 = st.columns(2)
            with col_prod1:
                num_products = st.selectbox(
                    "Nombre de produits",
                    options=[1, 2, 3, 4],
                    index=1,
                    help="Nombre de produits bancaires"
                )
            with col_prod2:
                has_credit_card = st.checkbox("Possède une carte de crédit", value=True)
            
            is_active_member = st.checkbox("Membre actif", value=True, help="Le client utilise régulièrement ses services")
        
        with col2:
            st.markdown("### 🔮 Résultat de Prédiction")
            
            # Conversion geography
            geography_germany = 1 if geography == "Germany" else 0
            geography_spain = 1 if geography == "Spain" else 0
            
            # Préparation des features
            features = {
                "CreditScore": credit_score,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": num_products,
                "HasCrCard": 1 if has_credit_card else 0,
                "IsActiveMember": 1 if is_active_member else 0,
                "EstimatedSalary": estimated_salary,
                "Geography_Germany": geography_germany,
                "Geography_Spain": geography_spain
            }
            
            # Bouton de prédiction
            if st.button("🔍 Analyser le Risque de Churn", type="primary", key="predict_btn"):
                with st.spinner("Analyse en cours..."):
                    result = make_prediction(features)
                    
                    if result:
                        # Affichage des métriques
                        col_m1, col_m2, col_m3 = st.columns(3)
                        
                        with col_m1:
                            st.metric(
                                label="Probabilité",
                                value=f"{result['churn_probability']*100:.1f}%"
                            )
                        
                        with col_m2:
                            pred_text = "🚪 Départ" if result['prediction'] == 1 else "✅ Fidèle"
                            st.metric(label="Prédiction", value=pred_text)
                        
                        with col_m3:
                            risk_colors = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
                            st.metric(
                                label="Risque",
                                value=f"{risk_colors.get(result['risk_level'], '')} {result['risk_level']}"
                            )
                        
                        # Graphique jauge
                        st.plotly_chart(create_gauge_chart(result['churn_probability']), use_container_width=True)
                        
                        # Recommandations
                        if result['risk_level'] == "High":
                            st.error("**⚠️ Risque élevé!** Contact prioritaire recommandé.")
                        elif result['risk_level'] == "Medium":
                            st.warning("**⚡ Risque modéré.** Surveillance recommandée.")
                        else:
                            st.success("**✅ Client stable.** Maintenir le suivi habituel.")
                        
                        st.session_state['last_prediction'] = result
            
            # Profil radar
            st.markdown("---")
            st.markdown("#### 📊 Profil Client")
            st.plotly_chart(create_feature_importance_chart(features), use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════════
    # TAB 2: DRIFT ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### 📊 Analyse du Drift des Données")
        st.markdown("Détectez les changements dans la distribution des données de production par rapport aux données d'entraînement.")
        
        col_drift1, col_drift2 = st.columns([1, 2])
        
        with col_drift1:
            st.markdown("#### ⚙️ Configuration")
            
            drift_intensity = st.slider(
                "Intensité du drift simulé",
                min_value=0.0, max_value=1.0, value=0.3, step=0.1,
                help="0 = pas de drift, 1 = drift maximum"
            )
            
            n_samples = st.slider(
                "Nombre d'échantillons",
                min_value=100, max_value=1000, value=500, step=100
            )
            
            if st.button("🔄 Générer & Analyser le Drift", type="primary", key="drift_btn"):
                with st.spinner("Génération des données et analyse du drift..."):
                    # Load reference data
                    try:
                        ref_data = pd.read_csv("https://raw.githubusercontent.com/AmelJaballah/bank-churn-mlops/main/data/bank_churn.csv")
                    except:
                        # Fallback: generate synthetic reference data
                        np.random.seed(42)
                        ref_data = pd.DataFrame({
                            'CreditScore': np.random.normal(650, 100, 1000),
                            'Age': np.random.normal(40, 12, 1000),
                            'Tenure': np.random.randint(0, 11, 1000),
                            'Balance': np.random.normal(75000, 50000, 1000),
                            'NumOfProducts': np.random.choice([1, 2, 3, 4], 1000, p=[0.5, 0.4, 0.08, 0.02]),
                            'EstimatedSalary': np.random.normal(100000, 50000, 1000)
                        })
                    
                    # Generate production data with drift
                    prod_data = generate_synthetic_drift_data(ref_data, drift_intensity)
                    
                    # Calculate drift metrics
                    numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
                    drift_results = {}
                    
                    for feature in numerical_features:
                        if feature in ref_data.columns:
                            ref_vals = ref_data[feature].dropna().values
                            prod_vals = prod_data[feature].dropna().values
                            
                            from scipy import stats
                            ks_stat, p_value = stats.ks_2samp(ref_vals, prod_vals)
                            psi = calculate_psi(ref_vals, prod_vals)
                            
                            drift_results[feature] = {
                                'ks_statistic': round(ks_stat, 4),
                                'p_value': round(p_value, 6),
                                'psi': round(psi, 4),
                                'drift_detected': psi > 0.1 or p_value < 0.05
                            }
                            
                            drift_results[feature] = {
                                'ks_statistic': round(ks_stat, 4),
                                'p_value': round(p_value, 6),
                                'psi': round(psi, 4),
                                'drift_detected': psi > 0.1 or p_value < 0.05
                            }
                    
                    # Store in session state
                    st.session_state['drift_results'] = drift_results
                    st.session_state['ref_data'] = ref_data
                    st.session_state['prod_data'] = prod_data
                    st.session_state['drift_intensity'] = drift_intensity
                    
                    st.success("✅ Analyse terminée!")
        
        with col_drift2:
            if 'drift_results' in st.session_state:
                drift_results = st.session_state['drift_results']
                
                # Summary metrics
                drifted = [f for f, r in drift_results.items() if r['drift_detected']]
                drift_pct = len(drifted) / len(drift_results) * 100
                
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("Features analysées", len(drift_results))
                with col_s2:
                    st.metric("Drift détecté", len(drifted))
                with col_s3:
                    status = "🟢 OK" if drift_pct < 30 else "🟡 Warning" if drift_pct < 60 else "🔴 Critical"
                    st.metric("Statut", status)
                
                # PSI Chart
                st.plotly_chart(create_drift_summary_chart(drift_results), use_container_width=True)
        
        # Distribution comparisons
        if 'drift_results' in st.session_state:
            st.markdown("---")
            st.markdown("#### 📈 Comparaison des Distributions")
            
            ref_data = st.session_state['ref_data']
            prod_data = st.session_state['prod_data']
            
            feature_cols = st.columns(3)
            features = ['CreditScore', 'Age', 'Balance']
            
            for i, feature in enumerate(features):
                with feature_cols[i]:
                    if feature in ref_data.columns:
                        st.plotly_chart(
                            create_drift_comparison_chart(ref_data, prod_data, feature),
                            use_container_width=True
                        )
            
            # Detailed table
            st.markdown("#### 📋 Détails par Feature")
            drift_df = pd.DataFrame(st.session_state['drift_results']).T
            drift_df['Status'] = drift_df['drift_detected'].apply(lambda x: '🔴 Drift' if x else '🟢 Stable')
            st.dataframe(drift_df[['psi', 'ks_statistic', 'p_value', 'Status']], use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════════
    # TAB 3: BATCH PREDICTION
    # ═══════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### 📁 Prédiction par Lot")
        st.info("Téléchargez un fichier CSV avec les colonnes: CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geography_Germany, Geography_Spain")
        
        uploaded_file = st.file_uploader("Choisir un fichier CSV", type=['csv'], key="batch_upload")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("**Aperçu des données:**")
                st.dataframe(df.head(), use_container_width=True)
                
                # Validate required columns
                required_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                                'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 
                                'Geography_Germany', 'Geography_Spain']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ Colonnes manquantes: {', '.join(missing_cols)}")
                else:
                    if st.button("🚀 Lancer l'analyse", type="primary", key="batch_btn"):
                        results = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for idx, row in df.iterrows():
                            # Ensure proper data types
                            feat = {
                                'CreditScore': int(row['CreditScore']),
                                'Age': int(row['Age']),
                                'Tenure': int(row['Tenure']),
                                'Balance': float(row['Balance']),
                                'NumOfProducts': int(row['NumOfProducts']),
                                'HasCrCard': int(row['HasCrCard']),
                                'IsActiveMember': int(row['IsActiveMember']),
                                'EstimatedSalary': float(row['EstimatedSalary']),
                                'Geography_Germany': int(row['Geography_Germany']),
                                'Geography_Spain': int(row['Geography_Spain'])
                            }
                            
                            result = make_prediction(feat)
                            if result:
                                results.append({**feat, **result})
                            
                            progress_bar.progress((idx + 1) / len(df))
                            status_text.text(f"Traitement: {idx + 1}/{len(df)}")
                        
                        status_text.empty()
                        
                        if results:
                            results_df = pd.DataFrame(results)
                            
                            col_b1, col_b2, col_b3, col_b4 = st.columns(4)
                            with col_b1:
                                st.metric("Total", len(results))
                            with col_b2:
                                high = len([r for r in results if r['risk_level'] == 'High'])
                                st.metric("Risque élevé", high)
                            with col_b3:
                                med = len([r for r in results if r['risk_level'] == 'Medium'])
                                st.metric("Risque modéré", med)
                            with col_b4:
                                churn_rate = sum([r['prediction'] for r in results]) / len(results) * 100
                                st.metric("Taux churn", f"{churn_rate:.1f}%")
                            
                            csv = results_df.to_csv(index=False)
                            st.download_button("📥 Télécharger résultats", csv, "predictions.csv", "text/csv")
                            
                            fig = px.histogram(results_df, x='churn_probability', nbins=20, color='risk_level',
                                              color_discrete_map={'Low': '#10b981', 'Medium': '#f59e0b', 'High': '#ef4444'})
                            fig.update_layout(title="Distribution des prédictions")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Aucune prédiction réussie.")
                        
            except Exception as e:
                st.error(f"Erreur: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>🏦 Bank Churn Prediction - MLOps Workshop 2026</p>
        <p>FastAPI • Streamlit • MLflow • Azure Container Apps</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
