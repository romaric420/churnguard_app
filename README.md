# ChurnGuard

**Système de Prédiction d'Attrition Client par Machine Learning**

[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python)](https://python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4.0-F7931E?logo=scikit-learn)](https://scikit-learn.org/)

---

## Description

**ChurnGuard** est une application de Data Science qui utilise des algorithmes de Machine Learning pour prédire le risque de départ des clients (churn). L'application permet d'identifier les clients à risque et de mettre en place des actions de rétention ciblées.

## Problématique Business

L'attrition client représente un coût majeur pour les entreprises :
- Acquérir un nouveau client coûte **5 à 25 fois plus cher** que retenir un client existant
- Une augmentation de **5% du taux de rétention** peut augmenter les profits de **25 à 95%**

ChurnGuard permet de :
-Identifier les clients à risque avant leur départ
-Comprendre les facteurs déterminants du churn
-Optimiser les budgets de rétention
-Améliorer la satisfaction client

---

## Structure du Projet

```
churnguard/
├── app.py                      # Page d'accueil
├── config.py                   # Configuration et constantes
├── data_loader.py              # Génération et chargement des données
├── requirements.txt            # Dépendances Python
├── README.md                   # Documentation
│
├── pages/                      # Pages de l'application
│   ├── 1_Dashboard.py          # Tableau de bord et KPIs
│   ├── 2_Exploration.py        # Analyse exploratoire
│   ├── 3_Modeles.py            # Performance des modèles ML
│   └── 4_Prediction.py         # Prédiction individuelle
│
├── utils/                      # Modules utilitaires
│   ├── __init__.py             # Package initialization
│   ├── models.py               # Fonctions ML
│   └── visualizations.py       # Graphiques Plotly
│
└── .streamlit/                 # Configuration Streamlit
    └── config.toml             # Thème personnalisé
```

---

## Fonctionnalités

### Dashboard
- Vue d'ensemble des métriques clés (KPIs)
- Taux de churn global et par segment
- Statistiques par type de contrat
- Insights automatiques

### Exploration
- Analyse univariée (variables catégorielles et continues)
- Analyse multivariée (churn vs features)
- Matrice de corrélation interactive
- Filtres dynamiques

### Modèles ML
- Comparaison de 3 algorithmes
- Métriques de performance (Accuracy, Precision, Recall, F1)
- Courbes ROC et AUC
- Matrices de confusion
- Validation croisée 5-fold
- Importance des variables

### Prédiction Individuelle
- Formulaire de saisie client
- Calcul du risque en temps réel
- Jauge de risque visuelle
- Identification des facteurs de risque
- Recommandations personnalisées

---

## Modèles Implémentés

| Modèle | Type | Description |
|--------|------|-------------|
| Régression Logistique | Classification | Modèle linéaire interprétable |
| KNN (k=5) | Classification | Approche basée sur les voisins proches |
| KNN (k=11) | Classification | Version stabilisée avec plus de voisins |

---

## Technologies

- **Python 3.9+** - Langage de programmation
- **Streamlit** - Framework d'application web
- **Scikit-learn** - Bibliothèque de Machine Learning
- **Pandas** - Manipulation de données
- **Plotly** - Visualisations interactives
- **NumPy** - Calculs numériques

---

## Installation

### Prérequis
- Python 3.9 ou supérieur
- pip (gestionnaire de packages Python)

### Installation locale

# j'ai utiliser Poetry et Pyenv
poetry : gestiion de l'environement et ajout des module
pyenv: gestion de la version de python 

```bash
# 1. Cloner ou extraire le projet
cd churnguard

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate      # Linux/Mac
# ou
venv\Scripts\activate         # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
```

L'application sera accessible sur **http://localhost:8501**

---

## Déploiement sur Streamlit Cloud

1. Créer un compte sur [share.streamlit.io](https://share.streamlit.io)
2. Connecter votre repository GitHub contenant le projet
3. Sélectionner le fichier `app.py` comme point d'entrée
4. Déployer

Streamlit détectera automatiquement le fichier `requirements.txt`.

---

## Variables du Dataset

| Variable | Type | Description |
|----------|------|-------------|
| customer_id | ID | Identifiant unique client |
| age | Continue | Âge du client |
| gender | Catégorielle | Genre (Homme/Femme) |
| tenure_months | Continue | Ancienneté en mois |
| monthly_charges | Continue | Charges mensuelles (€) |
| contract_type | Catégorielle | Type de contrat |
| payment_method | Catégorielle | Méthode de paiement |
| num_services | Discrète | Nombre de services souscrits |
| support_tickets | Discrète | Tickets support ouverts |
| satisfaction_score | Continue | Score de satisfaction (1-5) |
| **churn** | **Binaire** | **Variable cible (0/1)** |

---

## Auteur

**Romaric TCHOFFO**  
Projet Data Science - 2026

## Version 
1.0.0# churnguard_app
