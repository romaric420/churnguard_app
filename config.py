"""
ChurnGuard - Configuration et Constantes
=========================================
Fichier centralisant toutes les configurations de l'application
"""

# ═══════════════════════════════════════════════════════════════════════════════
# INFORMATIONS PROJET
# ═══════════════════════════════════════════════════════════════════════════════

APP_NAME = "ChurnGuard"
APP_ICON = "🛡️"
APP_DESCRIPTION = "Système de Prédiction d'Attrition Client par Machine Learning"
AUTHOR = "Romaric TCHOFFO"
VERSION = "1.0.0"

# ═══════════════════════════════════════════════════════════════════════════════
# COULEURS ET THEME
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'success': '#2ECC71',
    'warning': '#F39C12',
    'danger': '#E74C3C',
    'info': '#3498DB',
    'dark': '#1E3A5F',
    'light': '#f8f9fa',
    'churn': '#E94F37',
    'no_churn': '#2E86AB'
}

GRADIENT = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION DES DONNEES
# ═══════════════════════════════════════════════════════════════════════════════

N_SAMPLES = 5000
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Variables catégorielles
CATEGORICAL_COLUMNS = ['gender', 'contract_type', 'payment_method', 'online_activity']

# Variables numériques pour l'analyse
NUMERIC_COLUMNS = [
    'age', 'tenure_months', 'monthly_charges', 'total_charges',
    'num_services', 'support_tickets', 'satisfaction_score'
]

# Labels français pour les colonnes
COLUMN_LABELS = {
    'customer_id': 'ID Client',
    'age': 'Âge',
    'gender': 'Genre',
    'tenure_months': 'Ancienneté (mois)',
    'monthly_charges': 'Charges Mensuelles (€)',
    'total_charges': 'Charges Totales (€)',
    'contract_type': 'Type de Contrat',
    'payment_method': 'Méthode de Paiement',
    'num_services': 'Nombre de Services',
    'support_tickets': 'Tickets Support',
    'satisfaction_score': 'Score Satisfaction',
    'online_activity': 'Activité en Ligne',
    'has_partner': 'Partenaire',
    'has_dependents': 'Dépendants',
    'churn': 'Churn'
}

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION DES MODELES
# ═══════════════════════════════════════════════════════════════════════════════

MODELS_CONFIG = {
    'Régression Logistique': {
        'type': 'classification',
        'params': {'random_state': RANDOM_STATE, 'max_iter': 1000}
    },
    'KNN (k=5)': {
        'type': 'classification',
        'params': {'n_neighbors': 5}
    },
    'KNN (k=11)': {
        'type': 'classification',
        'params': {'n_neighbors': 11}
    }
}

# Features pour le ML
FEATURE_COLUMNS = [
    'age', 'tenure_months', 'monthly_charges', 'total_charges',
    'num_services', 'support_tickets', 'satisfaction_score',
    'has_partner', 'has_dependents',
    'gender_encoded', 'contract_type_encoded', 
    'payment_method_encoded', 'online_activity_encoded'
]

# ═══════════════════════════════════════════════════════════════════════════════
# CSS PERSONNALISE
# ═══════════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        padding: 1rem 0;
    }
    
    /* Sous-titre */
    .subtitle {
        color: #5A6C7D;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Cards métriques */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #667eea;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A5F;
    }
    
    .metric-label {
        color: #5A6C7D;
        font-size: 0.9rem;
    }
    
    /* Section header */
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
    }
    
    /* Insight card */
    .insight-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem 1.5rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
    
    .insight-card h4 {
        color: #1E3A5F;
        margin: 0 0 0.5rem 0;
    }
    
    .insight-card p {
        color: #5A6C7D;
        margin: 0;
    }
    
    /* Feature card */
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    /* Badge tech */
    .tech-badge {
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.85rem;
        margin-right: 0.5rem;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    /* Score cards */
    .score-excellent {
        background: linear-gradient(135deg, #2ECC71 0%, #27AE60 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
    }
    
    .score-good {
        background: linear-gradient(135deg, #3498DB 0%, #2980B9 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
    }
    
    .score-warning {
        background: linear-gradient(135deg, #F39C12 0%, #E67E22 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
    }
    
    .score-danger {
        background: linear-gradient(135deg, #E74C3C 0%, #C0392B 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom divider */
    .custom-divider {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
</style>
"""

# ═══════════════════════════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════════

def format_number(value, decimals=0):
    """Formate un nombre avec séparateurs de milliers"""
    if decimals == 0:
        return f"{int(value):,}".replace(",", " ")
    return f"{value:,.{decimals}f}".replace(",", " ")

def format_currency(value):
    """Formate un montant en euros"""
    return f"{format_number(value)} €"

def format_percentage(value):
    """Formate un pourcentage"""
    return f"{value:.1f}%"
