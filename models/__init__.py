"""
Models Package
==============
Stroke Prediction için kullanılan ML modelleri.

Modeller:
- Logistic Regression
- Random Forest
- XGBoost
- SVM
- KNN
"""

from .model_logistic import train_logistic_regression
from .model_random_forest import train_random_forest
from .model_svm import train_svm
from .model_knn import train_knn

# XGBoost opsiyonel
try:
    from .model_xgboost import train_xgboost
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

__all__ = [
    'train_logistic_regression',
    'train_random_forest', 
    'train_svm',
    'train_knn',
    'train_xgboost',
    'HAS_XGBOOST'
]
