"""
Prédiction Individuelle - ChurnGuard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="Prédiction - ChurnGuard", layout="wide")

# ============================================================================
# GÉNÉRATION DES DONNÉES ET MODÈLES
# ============================================================================

@st.cache_data
def load_data():
    np.random.seed(42)
    n = 5000
    
    data = {
        'customer_id': [f'CUST_{i:05d}' for i in range(n)],
        'age': np.random.normal(45, 15, n).clip(18, 80).astype(int),
        'gender': np.random.choice(['Homme', 'Femme'], n),
        'tenure_months': np.random.exponential(24, n).clip(1, 72).astype(int),
        'monthly_charges': np.random.normal(65, 30, n).clip(20, 150).round(2),
        'contract_type': np.random.choice(['Mensuel', 'Annuel', 'Bi-annuel'], n, p=[0.5, 0.3, 0.2]),
        'payment_method': np.random.choice(['Carte bancaire', 'Prélèvement', 'Virement', 'Chèque'], n, p=[0.4, 0.35, 0.15, 0.1]),
        'num_services': np.random.poisson(3, n).clip(1, 8),
        'support_tickets': np.random.poisson(2, n),
        'satisfaction_score': np.random.normal(3.5, 1, n).clip(1, 5).round(1),
        'online_activity': np.random.choice(['Faible', 'Moyenne', 'Élevée'], n),
        'has_partner': np.random.choice([0, 1], n, p=[0.4, 0.6]),
        'has_dependents': np.random.choice([0, 1], n, p=[0.7, 0.3])
    }
    
    df = pd.DataFrame(data)
    df['total_charges'] = (df['monthly_charges'] * df['tenure_months']).round(2)
    
    churn_prob = (
        0.1 +
        (df['contract_type'] == 'Mensuel').astype(float) * 0.25 +
        (df['tenure_months'] < 12).astype(float) * 0.15 +
        (df['support_tickets'] > 3).astype(float) * 0.2 +
        (df['satisfaction_score'] < 3).astype(float) * 0.25 +
        (df['monthly_charges'] > 80).astype(float) * 0.1 -
        (df['num_services'] > 4).astype(float) * 0.15
    ).clip(0.05, 0.85)
    
    df['churn'] = (np.random.random(n) < churn_prob).astype(int)
    
    return df

@st.cache_resource
def prepare_and_train():
    df = load_data()
    df_ml = df.copy()
    
    label_encoders = {}
    categorical_cols = ['gender', 'contract_type', 'payment_method', 'online_activity']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_ml[col + '_encoded'] = le.fit_transform(df_ml[col])
        label_encoders[col] = le
    
    feature_cols = [
        'age', 'tenure_months', 'monthly_charges', 'total_charges',
        'num_services', 'support_tickets', 'satisfaction_score',
        'has_partner', 'has_dependents',
        'gender_encoded', 'contract_type_encoded', 
        'payment_method_encoded', 'online_activity_encoded'
    ]
    
    X = df_ml[feature_cols]
    y = df_ml['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    models = {
        'Régression Logistique': LogisticRegression(random_state=42, max_iter=1000),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'KNN (k=11)': KNeighborsClassifier(n_neighbors=11)
    }
    
    for model in models.values():
        model.fit(X_train_scaled, y_train)
    
    return models, scaler, label_encoders

# ============================================================================
# PAGE
# ============================================================================

st.title("Prédiction Individuelle")
st.markdown("Estimez le risque de churn pour un client spécifique")

models, scaler, label_encoders = prepare_and_train()

# Sidebar
st.sidebar.header("Configuration")
selected_model = st.sidebar.selectbox("Modèle", list(models.keys()))

st.markdown("---")

# Formulaire
st.header("Caractéristiques du Client")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Profil")
    age = st.slider("Âge", 18, 80, 35)
    gender = st.selectbox("Genre", ['Homme', 'Femme'])
    has_partner = st.checkbox("A un partenaire")
    has_dependents = st.checkbox("A des dépendants")

with col2:
    st.subheader("Abonnement")
    tenure = st.slider("Ancienneté (mois)", 1, 72, 12)
    contract = st.selectbox("Type de contrat", ['Mensuel', 'Annuel', 'Bi-annuel'])
    monthly = st.number_input("Charges mensuelles (€)", 20.0, 150.0, 65.0)
    num_services = st.slider("Nombre de services", 1, 8, 3)

with col3:
    st.subheader("Engagement")
    payment = st.selectbox("Méthode de paiement", ['Carte bancaire', 'Prélèvement', 'Virement', 'Chèque'])
    activity = st.selectbox("Activité en ligne", ['Faible', 'Moyenne', 'Élevée'])
    tickets = st.slider("Tickets support", 0, 10, 2)
    satisfaction = st.slider("Score satisfaction", 1.0, 5.0, 3.5, 0.1)

st.markdown("---")

# Prédiction
if st.button("Analyser le Risque", type="primary", use_container_width=True):
    
    # Préparation
    new_data = pd.DataFrame({
        'age': [age],
        'tenure_months': [tenure],
        'monthly_charges': [monthly],
        'total_charges': [monthly * tenure],
        'num_services': [num_services],
        'support_tickets': [tickets],
        'satisfaction_score': [satisfaction],
        'has_partner': [1 if has_partner else 0],
        'has_dependents': [1 if has_dependents else 0],
        'gender_encoded': [label_encoders['gender'].transform([gender])[0]],
        'contract_type_encoded': [label_encoders['contract_type'].transform([contract])[0]],
        'payment_method_encoded': [label_encoders['payment_method'].transform([payment])[0]],
        'online_activity_encoded': [label_encoders['online_activity'].transform([activity])[0]]
    })
    
    features_scaled = scaler.transform(new_data)
    
    model = models[selected_model]
    prediction = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0][1] if hasattr(model, 'predict_proba') else prediction
    
    st.markdown("---")
    
    # Résultats
    st.header("Résultat")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if prediction == 1:
            st.error(f"## RISQUE ÉLEVÉ\n### Probabilité: {proba*100:.1f}%")
        else:
            st.success(f"## FAIBLE RISQUE\n### Probabilité: {proba*100:.1f}%")
        
        # Facteurs de risque
        st.subheader("Facteurs identifiés")
        
        risks = []
        if contract == 'Mensuel':
            risks.append("Contrat mensuel")
        if tenure < 12:
            risks.append("Client récent (< 12 mois)")
        if satisfaction < 3:
            risks.append("Satisfaction < 3")
        if tickets > 3:
            risks.append("Tickets support > 3")
        if monthly > 80:
            risks.append("Charges élevées")
        
        if risks:
            for r in risks:
                st.write(r)
        else:
            st.info("Aucun facteur de risque majeur")
    
    with col2:
        # Jauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            title={'text': "Score de Risque"},
            number={'suffix': '%'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#E94F37' if proba > 0.5 else '#2E86AB'},
                'steps': [
                    {'range': [0, 30], 'color': '#90EE90'},
                    {'range': [30, 60], 'color': '#FFD700'},
                    {'range': [60, 100], 'color': '#FF6B6B'}
                ],
                'threshold': {'line': {'color': 'red', 'width': 4}, 'thickness': 0.75, 'value': 50}
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Recommandations
    st.header("Recommandations")
    
    if prediction == 1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Actions Prioritaires:**
            - Contact proactif
            - Offre de fidélisation
            - Passage au contrat annuel
            """)
        with col2:
            st.markdown("""
            **Support:**
            - Conseiller dédié
            - Enquête satisfaction
            - Suivi personnalisé
            """)
    else:
        st.markdown("""
        **Actions de valorisation:**
        - Remercier sa fidélité
        - Proposer des services complémentaires
        - Programme parrainage
        """)
