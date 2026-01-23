"""
ChurnGuard - Module Modèles ML
==============================
Fonctions d'entraînement et d'évaluation des modèles
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import RANDOM_STATE, CATEGORICAL_COLUMNS, FEATURE_COLUMNS


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prépare les features pour le Machine Learning.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame des données brutes
        
    Returns
    -------
    tuple
        (X, y, label_encoders, feature_columns)
    """
    df_ml = df.copy()
    
    # Encodage des variables catégorielles
    label_encoders = {}
    
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        df_ml[col + '_encoded'] = le.fit_transform(df_ml[col])
        label_encoders[col] = le
    
    X = df_ml[FEATURE_COLUMNS]
    y = df_ml['churn']
    
    return X, y, label_encoders, FEATURE_COLUMNS


@st.cache_resource
def train_models(_X_train: pd.DataFrame, _y_train: pd.Series) -> tuple:
    """
    Entraîne les modèles de classification.
    
    Parameters
    ----------
    _X_train : pd.DataFrame
        Features d'entraînement
    _y_train : pd.Series
        Labels d'entraînement
        
    Returns
    -------
    tuple
        (trained_models, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(_X_train)
    
    models = {
        'Régression Logistique': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'KNN (k=11)': KNeighborsClassifier(n_neighbors=11)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, _y_train)
        trained_models[name] = model
    
    return trained_models, scaler


def evaluate_models(models: dict, X_test: pd.DataFrame, y_test: pd.Series, scaler: StandardScaler) -> pd.DataFrame:
    """
    Évalue tous les modèles sur le jeu de test.
    
    Parameters
    ----------
    models : dict
        Dictionnaire des modèles entraînés
    X_test : pd.DataFrame
        Features de test
    y_test : pd.Series
        Labels de test
    scaler : StandardScaler
        Scaler entraîné
        
    Returns
    -------
    pd.DataFrame
        DataFrame des résultats d'évaluation
    """
    X_test_scaled = scaler.transform(X_test)
    
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        results.append({
            'Modèle': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred)
        })
    
    return pd.DataFrame(results)


def get_roc_data(models: dict, X_test: pd.DataFrame, y_test: pd.Series, scaler: StandardScaler) -> dict:
    """
    Calcule les données ROC pour tous les modèles.
    
    Returns
    -------
    dict
        Données ROC par modèle
    """
    X_test_scaled = scaler.transform(X_test)
    
    roc_data = {}
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
    
    return roc_data


def get_confusion_matrix(model, X_test: pd.DataFrame, y_test: pd.Series, scaler: StandardScaler) -> np.ndarray:
    """
    Calcule la matrice de confusion.
    """
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    return confusion_matrix(y_test, y_pred)


def predict_single(model, scaler: StandardScaler, features: pd.DataFrame) -> tuple:
    """
    Prédiction pour un seul client.
    
    Returns
    -------
    tuple
        (prediction, probability)
    """
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features_scaled)[0][1]
    else:
        proba = float(prediction)
    
    return prediction, proba


def get_cross_validation_scores(models: dict, X: pd.DataFrame, y: pd.Series, scaler: StandardScaler, cv: int = 5) -> pd.DataFrame:
    """
    Calcule les scores de validation croisée.
    """
    X_scaled = scaler.transform(X)
    
    cv_results = []
    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1')
        cv_results.append({
            'Modèle': name,
            'F1 Moyen': scores.mean(),
            'Écart-type': scores.std(),
            'Min': scores.min(),
            'Max': scores.max()
        })
    
    return pd.DataFrame(cv_results)
