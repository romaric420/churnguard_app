"""
Exploration - ChurnGuard
===========================
Analyse exploratoire des données
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import CUSTOM_CSS, COLUMN_LABELS, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS
from data_loader import load_data
from utils.visualizations import (
    plot_churn_by_feature, plot_correlation_matrix, 
    plot_histogram, plot_boxplot
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Exploration - ChurnGuard", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<h1 class="main-header"> Exploration des Données</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analyse univariée et multivariée des facteurs de churn</p>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

df = load_data()

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR - FILTRES
# ═══════════════════════════════════════════════════════════════════════════════

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

st.sidebar.info(f"{len(df_filtered):,} clients analysés")

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSE UNIVARIÉE
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">Analyse Univariée</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Variables Catégorielles", "Variables Continues"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            plot_churn_by_feature(df_filtered, 'contract_type', "Churn par Type de Contrat"),
            use_container_width=True
        )
    
    with col2:
        st.plotly_chart(
            plot_churn_by_feature(df_filtered, 'payment_method', "Churn par Méthode de Paiement"),
            use_container_width=True
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            plot_churn_by_feature(df_filtered, 'online_activity', "Churn par Activité en Ligne"),
            use_container_width=True
        )
    
    with col2:
        st.plotly_chart(
            plot_churn_by_feature(df_filtered, 'gender', "Churn par Genre"),
            use_container_width=True
        )

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            plot_churn_by_feature(df_filtered, 'tenure_months', "Distribution de l'Ancienneté"),
            use_container_width=True
        )
    
    with col2:
        st.plotly_chart(
            plot_churn_by_feature(df_filtered, 'monthly_charges', "Distribution des Charges Mensuelles"),
            use_container_width=True
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            plot_churn_by_feature(df_filtered, 'satisfaction_score', "Distribution de la Satisfaction"),
            use_container_width=True
        )
    
    with col2:
        st.plotly_chart(
            plot_churn_by_feature(df_filtered, 'support_tickets', "Distribution des Tickets Support"),
            use_container_width=True
        )

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSE MULTIVARIÉE
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">Analyse Multivariée</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(
        plot_boxplot(df_filtered, 'monthly_charges', 'churn', "Charges Mensuelles par Statut Churn"),
        use_container_width=True
    )

with col2:
    st.plotly_chart(
        plot_boxplot(df_filtered, 'tenure_months', 'churn', "Ancienneté par Statut Churn"),
        use_container_width=True
    )

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MATRICE DE CORRÉLATION
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">Matrice de Corrélation</div>', unsafe_allow_html=True)

st.plotly_chart(plot_correlation_matrix(df_filtered), use_container_width=True)

st.markdown("""
<div class="insight-card">
    <h4>Comment lire la matrice ?</h4>
    <p>
    • Les valeurs proches de <strong>+1</strong> (bleu foncé) indiquent une corrélation positive forte<br>
    • Les valeurs proches de <strong>-1</strong> (rouge foncé) indiquent une corrélation négative forte<br>
    • Les valeurs proches de <strong>0</strong> indiquent peu ou pas de corrélation
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">Insights Clés de l\'Exploration</div>', unsafe_allow_html=True)

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
    - Charges mensuelles élevées
    """)

with col3:
    st.error("""
    **Facteurs de Risque**
    - Contrat mensuel
    - Satisfaction < 3
    - > 3 tickets support
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# STATISTIQUES DESCRIPTIVES
# ═══════════════════════════════════════════════════════════════════════════════

with st.expander("Statistiques Descriptives Complètes"):
    st.dataframe(df_filtered.describe().round(2), use_container_width=True)
