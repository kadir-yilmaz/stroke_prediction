"""
Random Forest Modeli
====================
Birden fazla karar ağacının birleşimi (ensemble).

Avantajları:
- Overfitting'e dayanıklı
- Feature importance verir
- Doğrusal olmayan ilişkileri yakalar
- Ölçekleme gerektirmez (ama yapılabilir)

Dezavantajları:
- Logistic Regression'a göre yavaş
- Yorumlanması daha zor
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import load_data_with_smote, print_results

def train_random_forest():
    # Veriyi yükle
    X_train, X_test, y_train, y_test, feature_names = load_data_with_smote()
    
    # Model oluştur
    model = RandomForestClassifier(
        n_estimators=100,        # 100 ağaç
        max_depth=10,            # Ağaç derinliği (overfitting önler)
        min_samples_split=5,     # Dallanma için min örnek
        min_samples_leaf=2,      # Yaprak için min örnek
        random_state=42,
        n_jobs=-1                # Tüm CPU'ları kullan
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
    results = print_results("RANDOM FOREST", y_test, y_pred, y_prob)
    
    # Feature Importance
    print("\n📊 En Önemli Özellikler (Feature Importance):")
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
    indices = np.argsort(importances)[-10:]  # Top 10
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], color='forestgreen')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest - En Önemli 10 Özellik')
    plt.tight_layout()
    plt.savefig('../results/rf_feature_importance.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    model, results = train_random_forest()
