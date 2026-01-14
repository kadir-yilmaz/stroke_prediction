"""
XGBoost Modeli
==============
Gradient Boosting tabanlı, yarışmalarda çok başarılı model.

Avantajları:
- Genellikle en iyi performans
- Regularization desteği
- Missing value handling
- Paralel işlem

Dezavantajları:
- Hyperparameter tuning gerektirir
- Overfitting riski (dikkatli tune edilmeli)
"""

try:
    from xgboost import XGBClassifier
except ImportError:
    print("⚠️ XGBoost yüklü değil. Yüklemek için: pip install xgboost")
    exit()

from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import load_data_with_smote, print_results

def train_xgboost():
    # Veriyi yükle
    X_train, X_test, y_train, y_test, feature_names = load_data_with_smote()
    
    # Model oluştur
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,              # Overfitting önlemek için sınırlı
        learning_rate=0.1,        # Öğrenme hızı
        subsample=0.8,            # Her ağaç için veri yüzdesi
        colsample_bytree=0.8,     # Her ağaç için özellik yüzdesi
        reg_alpha=0.1,            # L1 regularization
        reg_lambda=1.0,           # L2 regularization
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"\n🔄 5-Fold Cross-Validation ROC-AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Eğitim
    model.fit(X_train, y_train)
    
    # Tahmin
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Sonuçları yazdır
    results = print_results("XGBOOST", y_test, y_pred, y_prob)
    
    # Feature Importance
    print("\n📊 En Önemli Özellikler (XGBoost Importance):")
    importance = sorted(
        zip(feature_names, model.feature_importances_), 
        key=lambda x: x[1], 
        reverse=True
    )
    for feat, imp in importance[:5]:
        print(f"  • {feat}: {imp:.4f}")
    
    # Grafik
    plot_feature_importance(feature_names, model.feature_importances_)
    
    return model, results


def plot_feature_importance(feature_names, importances):
    """Feature importance grafiği çizer."""
    indices = np.argsort(importances)[-10:]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], color='darkorange')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('XGBoost - En Önemli 10 Özellik')
    plt.tight_layout()
    plt.savefig('../results/xgb_feature_importance.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    model, results = train_xgboost()
