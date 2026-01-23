"""
ChurnGuard - Package Utils
==========================
"""

from .models import (
    prepare_features,
    train_models,
    evaluate_models,
    get_roc_data,
    get_confusion_matrix,
    predict_single,
    get_cross_validation_scores
)

from .visualizations import (
    plot_churn_distribution,
    plot_churn_by_feature,
    plot_correlation_matrix,
    plot_roc_curves,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_risk_gauge,
    plot_histogram,
    plot_boxplot
)

__all__ = [
    # Models
    'prepare_features',
    'train_models',
    'evaluate_models',
    'get_roc_data',
    'get_confusion_matrix',
    'predict_single',
    'get_cross_validation_scores',
    # Visualizations
    'plot_churn_distribution',
    'plot_churn_by_feature',
    'plot_correlation_matrix',
    'plot_roc_curves',
    'plot_confusion_matrix',
    'plot_feature_importance',
    'plot_risk_gauge',
    'plot_histogram',
    'plot_boxplot'
]
