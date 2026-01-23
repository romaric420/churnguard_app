"""
ChurnGuard - Module Visualisations
==================================
Graphiques Plotly pour l'application
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import COLORS


def plot_churn_distribution(df: pd.DataFrame) -> go.Figure:
    """Graphique de distribution du churn (donut chart)"""
    churn_counts = df['churn'].value_counts()
    
    fig = go.Figure(data=[
        go.Pie(
            labels=['Clients fidèles', 'Clients partis'],
            values=[churn_counts.get(0, 0), churn_counts.get(1, 0)],
            hole=0.5,
            marker_colors=[COLORS['no_churn'], COLORS['churn']],
            textinfo='percent+value',
            textfont_size=14,
            pull=[0, 0.05]
        )
    ])
    
    fig.update_layout(
        title="Répartition du Churn",
        showlegend=True,
        height=400,
        font=dict(family="Arial, sans-serif")
    )
    
    return fig


def plot_churn_by_feature(df: pd.DataFrame, feature: str, title: str) -> go.Figure:
    """Analyse du churn par feature (bar chart ou histogram)"""
    
    if df[feature].dtype in ['int64', 'float64'] and df[feature].nunique() > 10:
        # Distribution continue
        fig = go.Figure()
        
        for churn_val, label, color in [(0, 'Fidèles', COLORS['no_churn']), 
                                         (1, 'Churn', COLORS['churn'])]:
            fig.add_trace(go.Histogram(
                x=df[df['churn'] == churn_val][feature],
                name=label,
                opacity=0.7,
                marker_color=color
            ))
        
        fig.update_layout(
            barmode='overlay',
            title=title,
            xaxis_title=feature,
            yaxis_title="Nombre de clients",
            height=400
        )
    else:
        # Taux de churn par catégorie
        churn_rate = df.groupby(feature)['churn'].agg(['mean', 'count']).reset_index()
        churn_rate.columns = [feature, 'taux_churn', 'count']
        churn_rate['taux_churn'] = churn_rate['taux_churn'] * 100
        
        fig = go.Figure(data=[
            go.Bar(
                x=churn_rate[feature],
                y=churn_rate['taux_churn'],
                marker_color=COLORS['primary'],
                text=churn_rate['taux_churn'].round(1).astype(str) + '%',
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title=feature,
            yaxis_title="Taux de Churn (%)",
            height=400
        )
    
    return fig


def plot_correlation_matrix(df: pd.DataFrame) -> go.Figure:
    """Matrice de corrélation heatmap"""
    numeric_cols = ['age', 'tenure_months', 'monthly_charges', 'num_services', 
                   'support_tickets', 'satisfaction_score', 'churn']
    corr_matrix = df[numeric_cols].corr()
    
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
    
    fig.update_layout(
        title="Matrice de Corrélation",
        height=500
    )
    
    return fig


def plot_roc_curves(roc_data: dict) -> go.Figure:
    """Courbes ROC comparatives"""
    fig = go.Figure()
    colors = [COLORS['primary'], COLORS['churn'], COLORS['warning']]
    
    for (name, data), color in zip(roc_data.items(), colors):
        fig.add_trace(go.Scatter(
            x=data['fpr'], 
            y=data['tpr'],
            name=f"{name} (AUC = {data['auc']:.3f})",
            line=dict(color=color, width=2)
        ))
    
    # Ligne de référence
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Référence (AUC = 0.5)',
        line=dict(color='gray', width=1, dash='dash')
    ))
    
    fig.update_layout(
        title="Courbes ROC - Comparaison des Modèles",
        xaxis_title="Taux de Faux Positifs (FPR)",
        yaxis_title="Taux de Vrais Positifs (TPR)",
        height=500,
        legend=dict(x=0.55, y=0.1)
    )
    
    return fig


def plot_confusion_matrix(cm: np.ndarray, model_name: str) -> go.Figure:
    """Matrice de confusion heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Prédit: Fidèle', 'Prédit: Churn'],
        y=['Réel: Fidèle', 'Réel: Churn'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 18},
        showscale=True
    ))
    
    fig.update_layout(
        title=f"Matrice de Confusion - {model_name}",
        height=400
    )
    
    return fig


def plot_feature_importance(model, feature_names: list) -> go.Figure:
    """Graphique d'importance des features (pour modèles linéaires)"""
    if not hasattr(model, 'coef_'):
        return None
    
    importance = np.abs(model.coef_[0])
    
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=df_imp['Importance'],
        y=df_imp['Feature'],
        orientation='h',
        marker_color=COLORS['primary']
    ))
    
    fig.update_layout(
        title="Importance des Variables (|coefficient|)",
        xaxis_title="Importance",
        yaxis_title="",
        height=500
    )
    
    return fig


def plot_risk_gauge(proba: float) -> go.Figure:
    """Jauge de risque de churn"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        title={'text': "Score de Risque", 'font': {'size': 20}},
        number={'suffix': '%', 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': COLORS['churn'] if proba > 0.5 else COLORS['no_churn']},
            'bgcolor': 'white',
            'steps': [
                {'range': [0, 30], 'color': '#90EE90'},
                {'range': [30, 60], 'color': '#FFD700'},
                {'range': [60, 100], 'color': '#FF6B6B'}
            ],
            'threshold': {
                'line': {'color': 'red', 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(height=300)
    
    return fig


def plot_histogram(df: pd.DataFrame, column: str, title: str, nbins: int = 30) -> go.Figure:
    """Histogramme simple"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df[column],
        nbinsx=nbins,
        marker_color=COLORS['primary'],
        opacity=0.8
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=column,
        yaxis_title="Fréquence",
        height=400
    )
    
    return fig


def plot_boxplot(df: pd.DataFrame, column: str, by: str = None, title: str = None) -> go.Figure:
    """Boxplot avec option de groupement"""
    if by:
        fig = px.box(df, x=by, y=column, color=by, title=title)
    else:
        fig = px.box(df, y=column, title=title)
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig
