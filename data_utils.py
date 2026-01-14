"""
Tüm modeller için paylaşılan veri yükleme ve ön işleme fonksiyonları.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Proje kök dizini
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FILEPATH = os.path.join(PROJECT_ROOT, "healthcare-dataset-stroke-data.csv")

def load_and_preprocess_data(filepath=None):
    """
    Veriyi yükler ve ön işleme yapar.
    
    Returns:
        X_train, X_test, y_train, y_test: Eğitim ve test verileri
        feature_names: Özellik isimleri
    """
    # 1. Veri Yükleme
    if filepath is None:
        filepath = DEFAULT_FILEPATH
    df = pd.read_csv(filepath)
    df.drop("id", axis=1, inplace=True)
    
    # 2. BMI eksik değerlerini doldur
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    df["bmi"].fillna(df["bmi"].median(), inplace=True)
    
    # 3. One-Hot Encoding (Label Encoding yerine - daha doğru)
    # Kategorik sütunlar sıralı olmadığı için One-Hot tercih edilir
    df = pd.get_dummies(df, columns=[
        "gender", "ever_married", "work_type", 
        "Residence_type", "smoking_status"
    ], drop_first=True)
    
    # 4. Özellik ve Hedef Ayrımı
    X = df.drop("stroke", axis=1)
    y = df["stroke"]
    feature_names = X.columns.tolist()
    
    # 5. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 6. Ölçekleme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names


def load_data_with_smote(filepath=None):
    """
    Veriyi yükler, SMOTE ile dengesizliği giderir.
    
    Dengesiz veri setlerinde azınlık sınıfını sentetik olarak çoğaltır.
    """
    # Önce normal şekilde yükle
    if filepath is None:
        filepath = DEFAULT_FILEPATH
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(filepath)
    
    # SMOTE uygula (sadece eğitim verisine!)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"SMOTE Öncesi: {sum(y_train==0)} sağlıklı, {sum(y_train==1)} felç")
    print(f"SMOTE Sonrası: {sum(y_train_resampled==0)} sağlıklı, {sum(y_train_resampled==1)} felç")
    
    return X_train_resampled, X_test, y_train_resampled, y_test, feature_names


def print_results(model_name, y_test, y_pred, y_prob=None):
    """
    Model sonuçlarını güzel bir şekilde yazdırır.
    """
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, precision_score, 
        recall_score, f1_score, classification_report
    )
    
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n{'='*50}")
    print(f"📊 {model_name} SONUÇLARI")
    print(f"{'='*50}")
    print(f"✅ Accuracy (Doğruluk)  : %{acc*100:.2f}")
    print(f"🎯 Precision (Kesinlik) : %{precision*100:.2f}")
    print(f"🔍 Recall (Yakalama)    : %{recall*100:.2f}")
    print(f"⚖️  F1-Score            : %{f1*100:.2f}")
    
    if y_prob is not None:
        auc = roc_auc_score(y_test, y_prob)
        print(f"📈 ROC-AUC             : {auc:.4f}")
    
    print(f"\n{'-'*50}")
    print("📋 Detaylı Rapor:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Sağlıklı (0)', 'Felç (1)']))
    
    return {
        'model': model_name,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc_score(y_test, y_prob) if y_prob is not None else None
    }
