"""
Analyse Exploratoire - ChurnGuard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Analyse - ChurnGuard", layout="wide")

  
# GÉNÉRATION DES DONNÉES
  

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

  
# PAGE
  

st.title("Analyse Exploratoire")
st.markdown("Exploration des facteurs de churn et relations entre variables")

df = load_data()

# Sidebar - Filtres
st.sidebar.header("Filtres")

contract_filter = st.sidebar.multiselect(
    "Type de contrat",
    df['contract_type'].unique(),
    default=list(df['contract_type'].unique())
)

churn_filter = st.sidebar.radio(
    "Statut client",
    ['Tous', 'Fidèles uniquement', 'Churn uniquement']
)

# Application des filtres
df_filtered = df[df['contract_type'].isin(contract_filter)]

if churn_filter == 'Fidèles uniquement':
    df_filtered = df_filtered[df_filtered['churn'] == 0]
elif churn_filter == 'Churn uniquement':
    df_filtered = df_filtered[df_filtered['churn'] == 1]

st.info(f"Analyse sur **{len(df_filtered):,}** clients")

st.markdown("---")

# Analyse par variable catégorielle
st.header("Analyse par Variable Catégorielle")

col1, col2 = st.columns(2)

with col1:
    # Churn par méthode de paiement
    churn_rate = df_filtered.groupby('payment_method')['churn'].mean() * 100
    fig = go.Figure(data=[go.Bar(
        x=churn_rate.index,
        y=churn_rate.values,
        marker_color='#667eea',
        text=churn_rate.round(1).astype(str) + '%',
        textposition='outside'
    )])
    fig.update_layout(title="Churn par Méthode de Paiement", yaxis_title="%", height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Churn par activité en ligne
    churn_rate = df_filtered.groupby('online_activity')['churn'].mean() * 100
    fig = go.Figure(data=[go.Bar(
        x=churn_rate.index,
        y=churn_rate.values,
        marker_color='#E94F37',
        text=churn_rate.round(1).astype(str) + '%',
        textposition='outside'
    )])
    fig.update_layout(title="Churn par Activité en Ligne", yaxis_title="%", height=400)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Analyse par variable continue
st.header("Analyse par Variable Continue")

col1, col2 = st.columns(2)

with col1:
    # Distribution ancienneté
    fig = go.Figure()
    for churn_val, label, color in [(0, 'Fidèles', '#2E86AB'), (1, 'Churn', '#E94F37')]:
        fig.add_trace(go.Histogram(
            x=df_filtered[df_filtered['churn'] == churn_val]['tenure_months'],
            name=label,
            opacity=0.7,
            marker_color=color
        ))
    fig.update_layout(barmode='overlay', title="Distribution de l'Ancienneté", height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Distribution satisfaction
    fig = go.Figure()
    for churn_val, label, color in [(0, 'Fidèles', '#2E86AB'), (1, 'Churn', '#E94F37')]:
        fig.add_trace(go.Histogram(
            x=df_filtered[df_filtered['churn'] == churn_val]['satisfaction_score'],
            name=label,
            opacity=0.7,
            marker_color=color
        ))
    fig.update_layout(barmode='overlay', title="Distribution de la Satisfaction", height=400)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Matrice de corrélation
st.header("Matrice de Corrélation")

numeric_cols = ['age', 'tenure_months', 'monthly_charges', 'num_services', 
               'support_tickets', 'satisfaction_score', 'churn']
corr_matrix = df_filtered[numeric_cols].corr()

fig = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    colorscale='RdBu',
    zmid=0,
    text=corr_matrix.values.round(2),
    texttemplate='%{text}',
    textfont={"size": 10}
))
fig.update_layout(title="Corrélation entre les Variables", height=500)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Insights
st.header("Insights Clés")

col1, col2, col3 = st.columns(3)

with col1:
    st.success("""
    **Facteurs Protecteurs**
    - Contrats longs (Annuel, Bi-annuel)
    - Ancienneté > 24 mois
    - Nombre de services > 4
    """)

with col2:
    st.warning("""
    **Facteurs à Surveiller**
    - Satisfaction entre 2.5 et 3.5
    - 1-2 tickets support récents
    """)

with col3:
    st.error("""
    **Facteurs de Risque**
    - Contrat mensuel
    - Satisfaction < 3
    - > 3 tickets support
    """)
