"""
Support Vector Machine (SVM) Modeli
===================================
Karar sınırı çizerek sınıfları ayıran güçlü model.

Avantajları:
- Yüksek boyutlu verilerde etkili
- Kernel trick ile doğrusal olmayan sınırlar
- Overfitting'e dayanıklı

Dezavantajları:
- Büyük veri setlerinde yavaş
- Ölçekleme şart!
- Probability estimation yavaş
"""

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import load_data_with_smote, print_results

def train_svm():
    # Veriyi yükle
    X_train, X_test, y_train, y_test, feature_names = load_data_with_smote()
    
    # Model oluştur
    # RBF kernel: Doğrusal olmayan sınırlar için
    model = SVC(
        kernel='rbf',             # Radial Basis Function
        C=1.0,                    # Regularization (küçük = daha fazla reg.)
        gamma='scale',            # Kernel katsayısı
        probability=True,         # predict_proba için gerekli
        random_state=42
    )
    
    # Cross-validation (SVM yavaş olduğu için cv=3)
    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc')
    print(f"\n🔄 3-Fold Cross-Validation ROC-AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Eğitim
    print("⏳ SVM eğitiliyor (biraz zaman alabilir)...")
    model.fit(X_train, y_train)
    
    # Tahmin
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Sonuçları yazdır
    results = print_results("SVM (RBF Kernel)", y_test, y_pred, y_prob)
    
    # SVM'de feature importance yok, support vector sayısını gösterelim
    print(f"\n📊 Support Vector Sayısı: {sum(model.n_support_)}")
    print(f"   - Class 0 (Sağlıklı): {model.n_support_[0]}")
    print(f"   - Class 1 (Felç): {model.n_support_[1]}")
    
    return model, results


if __name__ == "__main__":
    model, results = train_svm()
