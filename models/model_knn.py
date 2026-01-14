"""
K-Nearest Neighbors (KNN) Modeli
================================
En yakın k komşuya bakarak sınıflandırma yapar.

Avantajları:
- Basit ve sezgisel
- Eğitim aşaması yok (lazy learning)
- Non-parametric

Dezavantajları:
- Tahmin aşaması yavaş (tüm veriyi tarar)
- Yüksek boyutlarda performans düşer (curse of dimensionality)
- Ölçekleme şart!
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import load_data_with_smote, print_results

def find_best_k(X_train, y_train, k_range=range(1, 21)):
    """En iyi k değerini bulmak için test yapar."""
    k_scores = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='roc_auc')
        k_scores.append(scores.mean())
    
    best_k = k_range[np.argmax(k_scores)]
    print(f"🔍 En iyi k değeri: {best_k} (AUC: {max(k_scores):.4f})")
    
    # Grafik
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, k_scores, marker='o', color='purple')
    plt.xlabel('k (Komşu Sayısı)')
    plt.ylabel('Cross-Validation ROC-AUC')
    plt.title('KNN - k Değeri vs Performans')
    plt.xticks(k_range)
    plt.grid(True, alpha=0.3)
    plt.savefig('../results/knn_k_selection.png', dpi=150)
    plt.show()
    
    return best_k


def train_knn():
    # Veriyi yükle
    X_train, X_test, y_train, y_test, feature_names = load_data_with_smote()
    
    # En iyi k değerini bul
    best_k = find_best_k(X_train, y_train)
    
    # Model oluştur
    model = KNeighborsClassifier(
        n_neighbors=best_k,
        weights='distance',       # Yakın komşulara daha fazla ağırlık
        metric='euclidean',       # Mesafe metriği
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
    results = print_results(f"KNN (k={best_k})", y_test, y_pred, y_prob)
    
    return model, results


if __name__ == "__main__":
    model, results = train_knn()
