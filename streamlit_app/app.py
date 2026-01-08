"""
Bank Churn Prediction - Streamlit Interface
Application web pour la prédiction de défaillance client (churn)
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any
import os

# Configuration de la page
st.set_page_config(
    page_title="Bank Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL de l'API (variable d'environnement ou valeur par défaut)
API_URL = os.getenv("API_URL", "https://bank-churn.salmonbay-9a939e1f.francecentral.azurecontainerapps.io")

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-low {
        background-color: #28a745;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #ffc107;
        color: black;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
    }
    .risk-high {
        background-color: #dc3545;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.1rem;
        padding: 0.75rem 1rem;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #155a8a;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health() -> bool:
    """Vérifie si l'API est disponible"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def make_prediction(features: Dict[str, Any]) -> Dict[str, Any]:
    """Envoie une requête de prédiction à l'API"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=features,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API: {e}")
        return None

def create_gauge_chart(probability: float) -> go.Figure:
    """Crée un graphique jauge pour la probabilité de churn"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Probabilité de Churn (%)", 'font': {'size': 20}},
        delta={'reference': 20, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#28a745'},
                {'range': [30, 60], 'color': '#ffc107'},
                {'range': [60, 100], 'color': '#dc3545'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def create_feature_importance_chart(features: Dict[str, Any]) -> go.Figure:
    """Crée un graphique des features du client"""
    # Normalisation simple pour visualisation
    normalized = {
        'Credit Score': features['CreditScore'] / 850 * 100,
        'Âge': features['Age'] / 100 * 100,
        'Ancienneté': features['Tenure'] / 10 * 100,
        'Solde (norm)': min(features['Balance'] / 200000 * 100, 100),
        'Nb Produits': features['NumOfProducts'] / 4 * 100,
        'Carte Crédit': features['HasCrCard'] * 100,
        'Membre Actif': features['IsActiveMember'] * 100,
        'Salaire (norm)': min(features['EstimatedSalary'] / 150000 * 100, 100),
    }
    
    fig = go.Figure(go.Bar(
        x=list(normalized.values()),
        y=list(normalized.keys()),
        orientation='h',
        marker_color=['#1f77b4'] * len(normalized)
    ))
    fig.update_layout(
        title="Profil Client (valeurs normalisées)",
        xaxis_title="Score normalisé (%)",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🏦 Bank Churn Prediction</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Vérification de l'état de l'API
    with st.sidebar:
        st.markdown("### 🔗 État du Système")
        api_status = check_api_health()
        if api_status:
            st.success("✅ API Connectée")
        else:
            st.error("❌ API Non Disponible")
        
        st.markdown("---")
        st.markdown("### ℹ️ À propos")
        st.info("""
        Cette application utilise un modèle de Machine Learning 
        pour prédire la probabilité qu'un client quitte la banque.
        
        **Modèle:** Random Forest Classifier  
        **Précision:** ~85%
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Légende des Risques")
        st.markdown("🟢 **Low**: < 30%")
        st.markdown("🟡 **Medium**: 30-60%")
        st.markdown("🔴 **High**: > 60%")
    
    # Formulaire principal
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📋 Informations Client")
        
        # Informations personnelles
        st.markdown("#### 👤 Profil Personnel")
        age = st.slider("Âge", min_value=18, max_value=100, value=35, help="Âge du client")
        
        col_geo1, col_geo2 = st.columns(2)
        with col_geo1:
            geography = st.selectbox(
                "Pays",
                options=["France", "Germany", "Spain"],
                help="Pays de résidence du client"
            )
        with col_geo2:
            tenure = st.slider("Ancienneté (années)", min_value=0, max_value=10, value=5)
        
        # Informations financières
        st.markdown("#### 💰 Informations Financières")
        credit_score = st.slider(
            "Score de Crédit",
            min_value=300, max_value=850, value=650,
            help="Score de crédit (300-850)"
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
        if st.button("🔍 Analyser le Risque de Churn", type="primary"):
            with st.spinner("Analyse en cours..."):
                result = make_prediction(features)
                
                if result:
                    # Affichage des métriques
                    col_m1, col_m2, col_m3 = st.columns(3)
                    
                    with col_m1:
                        st.metric(
                            label="Probabilité",
                            value=f"{result['churn_probability']*100:.1f}%",
                            delta=None
                        )
                    
                    with col_m2:
                        pred_text = "🚪 Risque de départ" if result['prediction'] == 1 else "✅ Client fidèle"
                        st.metric(
                            label="Prédiction",
                            value=pred_text
                        )
                    
                    with col_m3:
                        risk_colors = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
                        st.metric(
                            label="Niveau de Risque",
                            value=f"{risk_colors.get(result['risk_level'], '')} {result['risk_level']}"
                        )
                    
                    # Graphique jauge
                    st.plotly_chart(
                        create_gauge_chart(result['churn_probability']),
                        use_container_width=True
                    )
                    
                    # Recommandations basées sur le risque
                    st.markdown("### 💡 Recommandations")
                    if result['risk_level'] == "High":
                        st.error("""
                        **⚠️ Risque élevé de churn détecté!**
                        
                        Actions recommandées:
                        - 📞 Contact prioritaire par un conseiller
                        - 🎁 Proposer des offres de fidélisation
                        - 📊 Analyser les réclamations récentes
                        - 💳 Revoir les conditions tarifaires
                        """)
                    elif result['risk_level'] == "Medium":
                        st.warning("""
                        **⚡ Risque modéré - Surveillance recommandée**
                        
                        Actions recommandées:
                        - 📧 Envoyer une enquête de satisfaction
                        - 🏷️ Proposer des produits complémentaires
                        - 📅 Planifier un point de contact
                        """)
                    else:
                        st.success("""
                        **✅ Client à faible risque**
                        
                        Actions recommandées:
                        - 📧 Maintenir une communication régulière
                        - 🌟 Programme de parrainage
                        - 📈 Proposer des produits d'investissement
                        """)
                    
                    # Sauvegarder dans session state
                    st.session_state['last_prediction'] = result
                    st.session_state['last_features'] = features
        
        # Afficher le profil client
        st.markdown("---")
        st.plotly_chart(
            create_feature_importance_chart(features),
            use_container_width=True
        )
    
    # Section batch prediction
    st.markdown("---")
    st.markdown("### 📁 Prédiction par Lot (Batch)")
    
    with st.expander("📤 Télécharger un fichier CSV pour analyse multiple"):
        st.info("""
        Format attendu du fichier CSV:
        - CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geography_Germany, Geography_Spain
        """)
        
        uploaded_file = st.file_uploader("Choisir un fichier CSV", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Aperçu des données:")
                st.dataframe(df.head())
                
                if st.button("🚀 Lancer l'analyse batch"):
                    results = []
                    progress_bar = st.progress(0)
                    
                    for idx, row in df.iterrows():
                        features = row.to_dict()
                        result = make_prediction(features)
                        if result:
                            results.append({
                                **features,
                                'churn_probability': result['churn_probability'],
                                'prediction': result['prediction'],
                                'risk_level': result['risk_level']
                            })
                        progress_bar.progress((idx + 1) / len(df))
                    
                    if results:
                        results_df = pd.DataFrame(results)
                        st.success(f"✅ {len(results)} prédictions effectuées!")
                        
                        # Statistiques
                        col_s1, col_s2, col_s3 = st.columns(3)
                        with col_s1:
                            st.metric("Total clients", len(results))
                        with col_s2:
                            high_risk = len([r for r in results if r['risk_level'] == 'High'])
                            st.metric("Risque élevé", high_risk)
                        with col_s3:
                            churn_rate = sum([r['prediction'] for r in results]) / len(results) * 100
                            st.metric("Taux de churn prédit", f"{churn_rate:.1f}%")
                        
                        # Télécharger les résultats
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "📥 Télécharger les résultats",
                            csv,
                            "predictions.csv",
                            "text/csv"
                        )
                        
                        # Graphique de distribution
                        fig = px.histogram(
                            results_df, x='churn_probability',
                            nbins=20,
                            title="Distribution des probabilités de churn",
                            labels={'churn_probability': 'Probabilité de Churn'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Erreur lors de la lecture du fichier: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🏦 Bank Churn Prediction - MLOps Workshop</p>
        <p>Powered by FastAPI + Streamlit + Azure Container Apps</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
