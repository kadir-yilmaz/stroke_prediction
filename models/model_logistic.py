"""
Logistic Regression Modeli
==========================
Binary classification için temel ve yorumlanabilir model.

Avantajları:
- Hızlı eğitim
- Katsayılar yorumlanabilir
- Küçük veri setlerinde iyi çalışır

Dezavantajları:
- Doğrusal olmayan ilişkileri yakalayamaz
- Feature engineering gerektirir
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import load_data_with_smote, print_results

def train_logistic_regression():
    # Veriyi yükle (SMOTE ile dengelenmiş)
    X_train, X_test, y_train, y_test, feature_names = load_data_with_smote()
    
    # Model oluştur
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='liblinear',
        C=1.0  # Regularization parametresi
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
    results = print_results("LOGISTIC REGRESSION", y_test, y_pred, y_prob)
    
    # Katsayıları göster (en etkili özellikler)
    print("\n📊 En Etkili Özellikler (Katsayı Büyüklüğüne Göre):")
    coef_importance = sorted(
        zip(feature_names, model.coef_[0]), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )
    for feat, coef in coef_importance[:5]:
        direction = "↑ Risk Artırıcı" if coef > 0 else "↓ Risk Azaltıcı"
        print(f"  • {feat}: {coef:.4f} ({direction})")
    
    return model, results


if __name__ == "__main__":
    model, results = train_logistic_regression()
