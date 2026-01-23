"""
Dashboard - ChurnGuard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Dashboard - ChurnGuard", layout="wide")

# ============================================================================
# GÉNÉRATION DES DONNÉES
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
    
    # Logique de churn
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

# ============================================================================
# PAGE
# ============================================================================

st.title("Dashboard")
st.markdown("Vue d'ensemble des données clients et indicateurs clés")

df = load_data()

# KPIs
st.header("Indicateurs Clés")

col1, col2, col3, col4 = st.columns(4)

churn_rate = df['churn'].mean() * 100

with col1:
    st.metric("Total Clients", f"{len(df):,}")

with col2:
    st.metric("Taux de Churn", f"{churn_rate:.1f}%", delta=f"{churn_rate - 15:.1f}% vs objectif", delta_color="inverse")

with col3:
    st.metric("Ancienneté Moyenne", f"{df['tenure_months'].mean():.0f} mois")

with col4:
    st.metric("Charges Moyennes", f"{df['monthly_charges'].mean():.0f} €/mois")

st.markdown("---")

# Graphiques
col1, col2 = st.columns(2)

with col1:
    # Pie chart churn
    churn_counts = df['churn'].value_counts()
    fig = go.Figure(data=[go.Pie(
        labels=['Fidèles', 'Churn'],
        values=[churn_counts.get(0, 0), churn_counts.get(1, 0)],
        hole=0.5,
        marker_colors=['#2E86AB', '#E94F37']
    )])
    fig.update_layout(title="Répartition du Churn", height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Churn par contrat
    churn_by_contract = df.groupby('contract_type')['churn'].mean() * 100
    fig = go.Figure(data=[go.Bar(
        x=churn_by_contract.index,
        y=churn_by_contract.values,
        marker_color='#667eea',
        text=churn_by_contract.round(1).astype(str) + '%',
        textposition='outside'
    )])
    fig.update_layout(title="Taux de Churn par Type de Contrat", yaxis_title="%", height=400)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Stats par segment
st.header("Statistiques par Segment")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Par Type de Contrat")
    contract_stats = df.groupby('contract_type').agg({
        'customer_id': 'count',
        'churn': 'mean',
        'monthly_charges': 'mean'
    }).round(2)
    contract_stats.columns = ['Nombre', 'Taux Churn', 'Charges Moy.']
    contract_stats['Taux Churn'] = (contract_stats['Taux Churn'] * 100).round(1).astype(str) + '%'
    st.dataframe(contract_stats, use_container_width=True)

with col2:
    st.subheader("Insights Clés")
    
    mensuel_churn = df[df['contract_type'] == 'Mensuel']['churn'].mean() * 100
    low_sat_churn = df[df['satisfaction_score'] < 3]['churn'].mean() * 100
    
    st.warning(f"Contrats Mensuels : {mensuel_churn:.1f}% de churn")
    st.error(f"Satisfaction < 3 : {low_sat_churn:.1f}% de churn")

# Données brutes
with st.expander("Voir les données brutes"):
    st.dataframe(df.head(100), use_container_width=True)
