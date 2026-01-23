"""
ChurnGuard - Data Loader
========================
Module de génération et chargement des données
"""

import pandas as pd
import numpy as np
import streamlit as st
from config import N_SAMPLES, RANDOM_STATE, CATEGORICAL_COLUMNS


@st.cache_data
def generate_churn_data(n_samples: int = N_SAMPLES) -> pd.DataFrame:
    """
    Génère des données synthétiques réalistes de churn client.
    
    Parameters
    ----------
    n_samples : int
        Nombre d'échantillons à générer
        
    Returns
    -------
    pd.DataFrame
        DataFrame contenant les données clients
    """
    np.random.seed(RANDOM_STATE)
    
    # Génération des données de base
    data = {
        'customer_id': [f'CUST_{i:05d}' for i in range(n_samples)],
        'age': np.random.normal(45, 15, n_samples).clip(18, 80).astype(int),
        'gender': np.random.choice(['Homme', 'Femme'], n_samples),
        'tenure_months': np.random.exponential(24, n_samples).clip(1, 72).astype(int),
        'monthly_charges': np.random.normal(65, 30, n_samples).clip(20, 150).round(2),
        'total_charges': np.zeros(n_samples),
        'contract_type': np.random.choice(
            ['Mensuel', 'Annuel', 'Bi-annuel'], 
            n_samples, 
            p=[0.5, 0.3, 0.2]
        ),
        'payment_method': np.random.choice(
            ['Carte bancaire', 'Prélèvement', 'Virement', 'Chèque'],
            n_samples,
            p=[0.4, 0.35, 0.15, 0.1]
        ),
        'num_services': np.random.poisson(3, n_samples).clip(1, 8),
        'support_tickets': np.random.poisson(2, n_samples),
        'satisfaction_score': np.random.normal(3.5, 1, n_samples).clip(1, 5).round(1),
        'online_activity': np.random.choice(['Faible', 'Moyenne', 'Élevée'], n_samples),
        'has_partner': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'has_dependents': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    df = pd.DataFrame(data)
    
    # Calcul des charges totales
    df['total_charges'] = (df['monthly_charges'] * df['tenure_months']).round(2)
    
    # Logique de churn basée sur les features
    churn_prob = (
        0.1 +
        (df['contract_type'] == 'Mensuel').astype(float) * 0.25 +
        (df['tenure_months'] < 12).astype(float) * 0.15 +
        (df['support_tickets'] > 3).astype(float) * 0.2 +
        (df['satisfaction_score'] < 3).astype(float) * 0.25 +
        (df['monthly_charges'] > 80).astype(float) * 0.1 -
        (df['num_services'] > 4).astype(float) * 0.15
    ).clip(0.05, 0.85)
    
    df['churn'] = (np.random.random(n_samples) < churn_prob).astype(int)
    
    return df


@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Charge les données (génération ou fichier CSV).
    
    Returns
    -------
    pd.DataFrame
        DataFrame des données clients
    """
    return generate_churn_data()


def get_summary_stats(df: pd.DataFrame) -> dict:
    """
    Calcule les statistiques résumées du dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame des données
        
    Returns
    -------
    dict
        Dictionnaire des statistiques
    """
    return {
        'total_clients': len(df),
        'churn_count': df['churn'].sum(),
        'churn_rate': df['churn'].mean() * 100,
        'avg_tenure': df['tenure_months'].mean(),
        'avg_charges': df['monthly_charges'].mean(),
        'avg_satisfaction': df['satisfaction_score'].mean(),
        'total_revenue': df['total_charges'].sum()
    }


def get_churn_by_category(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Calcule le taux de churn par catégorie.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame des données
    column : str
        Colonne catégorielle à analyser
        
    Returns
    -------
    pd.DataFrame
        DataFrame avec les statistiques par catégorie
    """
    return df.groupby(column).agg({
        'customer_id': 'count',
        'churn': ['sum', 'mean']
    }).round(3)
