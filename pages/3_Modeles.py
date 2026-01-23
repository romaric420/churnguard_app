"""
Modèles ML - ChurnGuard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

st.set_page_config(page_title="Modèles - ChurnGuard", layout="wide")

  
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

@st.cache_resource
def train_models(_X_train, _y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(_X_train)
    
    models = {
        'Régression Logistique': LogisticRegression(random_state=42, max_iter=1000),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'KNN (k=11)': KNeighborsClassifier(n_neighbors=11)
    }
    
    trained = {}
    for name, model in models.items():
        model.fit(X_train_scaled, _y_train)
        trained[name] = model
    
    return trained, scaler

  
# PAGE
  

st.title("Modèles de Machine Learning")
st.markdown("Entraînement, évaluation et comparaison des modèles")

df = load_data()

# Préparation des features
df_ml = df.copy()
categorical_cols = ['gender', 'contract_type', 'payment_method', 'online_activity']

for col in categorical_cols:
    le = LabelEncoder()
    df_ml[col + '_encoded'] = le.fit_transform(df_ml[col])

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

# Entraînement
models, scaler = train_models(X_train, y_train)

# Sidebar
st.sidebar.header("Configuration")
selected_model = st.sidebar.selectbox("Modèle à analyser", list(models.keys()))

st.markdown("---")

# Résultats
st.header("Comparaison des Performances")

X_test_scaled = scaler.transform(X_test)

results = []
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    results.append({
        'Modèle': name,
        'Accuracy': f"{accuracy_score(y_test, y_pred)*100:.2f}%",
        'Precision': f"{precision_score(y_test, y_pred)*100:.2f}%",
        'Recall': f"{recall_score(y_test, y_pred)*100:.2f}%",
        'F1-Score': f"{f1_score(y_test, y_pred)*100:.2f}%"
    })

st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

st.markdown("---")

# Graphiques
col1, col2 = st.columns(2)

with col1:
    # Courbes ROC
    st.subheader("Courbes ROC")
    
    fig = go.Figure()
    colors = ['#667eea', '#E94F37', '#F39C12']
    
    for (name, model), color in zip(models.items(), colors):
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={roc_auc:.3f})", line=dict(color=color)))
    
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Référence', line=dict(color='gray', dash='dash')))
    fig.update_layout(xaxis_title="FPR", yaxis_title="TPR", height=450)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Matrice de confusion
    st.subheader(f"Matrice de Confusion - {selected_model}")
    
    y_pred = models[selected_model].predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Prédit: Fidèle', 'Prédit: Churn'],
        y=['Réel: Fidèle', 'Réel: Churn'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 18}
    ))
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Importance des features (pour LogReg)
if selected_model == 'Régression Logistique':
    st.header("Importance des Variables")
    
    importance = np.abs(models[selected_model].coef_[0])
    df_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': importance}).sort_values('Importance', ascending=True)
    
    fig = go.Figure(go.Bar(x=df_imp['Importance'], y=df_imp['Feature'], orientation='h', marker_color='#667eea'))
    fig.update_layout(title="Importance (|coefficient|)", height=500)
    st.plotly_chart(fig, use_container_width=True)

# Validation croisée
st.header("Validation Croisée (5-Fold)")

X_scaled = scaler.transform(X)
cv_results = []

for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='f1')
    cv_results.append({
        'Modèle': name,
        'F1 Moyen': f"{scores.mean()*100:.2f}%",
        'Écart-type': f"{scores.std()*100:.2f}%"
    })

st.dataframe(pd.DataFrame(cv_results), use_container_width=True, hide_index=True)
