"""
5 ML modelini eğitir, kaydeder ve detaylı sonuçları results/ klasörüne yazar.
"""

import os
import sys
import time
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
import joblib

# Proje dizinleri
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def print_header(text):
    """Başlık yazdırır."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def load_and_prepare_data():
    """Veriyi yükler ve hazırlar."""
    print_header("VERİ YÜKLEME VE HAZIRLAMA")
    
    filepath = os.path.join(PROJECT_ROOT, "healthcare-dataset-stroke-data.csv")
    df = pd.read_csv(filepath)
    print(f"✅ Veri yüklendi: {len(df)} kayıt")
    
    # id kolonunu kaldır
    df.drop("id", axis=1, inplace=True)
    
    # BMI eksik değerleri doldur
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    bmi_median = df["bmi"].median()
    df["bmi"] = df["bmi"].fillna(bmi_median)
    print(f"✅ BMI eksik değerleri dolduruldu (median: {bmi_median:.1f})")
    
    # Kategorik değişkenleri encode et
    df = pd.get_dummies(df, columns=[
        "gender", "ever_married", "work_type", 
        "Residence_type", "smoking_status"
    ], drop_first=True)
    
    X = df.drop("stroke", axis=1)
    y = df["stroke"]
    
    print(f"✅ Özellik sayısı: {X.shape[1]}")
    print(f"✅ Hedef dağılımı: Sağlıklı={sum(y==0)}, Felç={sum(y==1)} ({sum(y==1)/len(y)*100:.1f}%)")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # SMOTE ile dengeleme
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print(f"✅ SMOTE sonrası train: {len(X_train_resampled)} (dengeli)")
    
    # Scaler ve feature columns kaydet
    joblib.dump(scaler, os.path.join(SAVED_MODELS_DIR, "scaler.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(SAVED_MODELS_DIR, "feature_columns.pkl"))
    print(f"✅ Scaler ve feature columns kaydedildi")
    
    return X_train_resampled, X_test_scaled, y_train_resampled, y_test, X.columns.tolist()


def get_feature_importance(model, feature_columns):
    """Model için feature importance çıkarır."""
    if hasattr(model, 'coef_'):
        coefs = np.abs(model.coef_[0])
        return list(zip(feature_columns, coefs))
    elif hasattr(model, 'feature_importances_'):
        return list(zip(feature_columns, model.feature_importances_))
    return None


def train_model(name, model, X_train, X_test, y_train, y_test, feature_columns):
    """Tek bir modeli eğitir ve metriklerini döner."""
    print(f"\n🔄 {name} eğitiliyor...")
    start_time = time.time()
    
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Kaydet
    model_file = f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
    joblib.dump(model, os.path.join(SAVED_MODELS_DIR, model_file))
    
    # Tahmin
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Metrikler
    metrics = {
        'model': name,
        'filename': model_file,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob) if y_prob is not None else 0,
        'train_time': train_time,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'feature_importance': get_feature_importance(model, feature_columns)
    }
    
    print(f"   ✅ {name} tamamlandı ({train_time:.2f}s)")
    print(f"      Accuracy: {metrics['accuracy']*100:.2f}%  |  Recall: {metrics['recall']*100:.2f}%  |  AUC: {metrics['roc_auc']:.4f}")
    
    return metrics


def train_all_models(X_train, X_test, y_train, y_test, feature_columns):
    """Tüm modelleri eğitir."""
    print_header("MODEL EĞİTİMİ")
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    
    models = [
        ("Logistic Regression", LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')),
        ("Random Forest", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
        ("SVM (RBF)", SVC(kernel='rbf', probability=True, random_state=42)),
        ("KNN (k=5)", KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)),
    ]
    
    # XGBoost ekle (varsa)
    try:
        from xgboost import XGBClassifier
        models.insert(2, ("XGBoost", XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42, eval_metric='logloss', n_jobs=-1
        )))
    except ImportError:
        print("⚠️ XGBoost yüklü değil, atlanıyor...")
    
    results = []
    for name, model in models:
        metrics = train_model(name, model, X_train, X_test, y_train, y_test, feature_columns)
        results.append(metrics)
    
    return results


def write_results(results, feature_columns):
    """Sonuçları dosyalara yazar."""
    print_header("SONUÇLARI KAYDETME")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. TXT Rapor
    report_path = os.path.join(RESULTS_DIR, "training_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("         STROKE PREDICTION - MODEL EĞİTİM RAPORU\n")
        f.write("=" * 70 + "\n")
        f.write(f"Tarih: {timestamp}\n\n")
        
        # Karşılaştırma tablosu
        f.write("-" * 70 + "\n")
        f.write("MODEL KARŞILAŞTIRMA TABLOSU\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}\n")
        f.write("-" * 70 + "\n")
        
        sorted_results = sorted(results, key=lambda x: x['roc_auc'], reverse=True)
        for r in sorted_results:
            f.write(f"{r['model']:<25} {r['accuracy']*100:>9.2f}% {r['precision']*100:>9.2f}% {r['recall']*100:>9.2f}% {r['f1']*100:>9.2f}% {r['roc_auc']:>10.4f}\n")
        
        f.write("\n")
        
        # Her model için detay
        for r in results:
            f.write("=" * 70 + "\n")
            f.write(f"{r['model']}\n")
            f.write("=" * 70 + "\n")
            f.write(f"Dosya: saved_models/{r['filename']}\n")
            f.write(f"Eğitim Süresi: {r['train_time']:.2f} saniye\n\n")
            
            f.write("Metrikler:\n")
            f.write(f"  - Accuracy:  {r['accuracy']*100:.2f}%\n")
            f.write(f"  - Precision: {r['precision']*100:.2f}%\n")
            f.write(f"  - Recall:    {r['recall']*100:.2f}%\n")
            f.write(f"  - F1-Score:  {r['f1']*100:.2f}%\n")
            f.write(f"  - ROC-AUC:   {r['roc_auc']:.4f}\n\n")
            
            f.write("Confusion Matrix:\n")
            cm = r['confusion_matrix']
            f.write(f"  TN={cm[0][0]:4d}  FP={cm[0][1]:4d}\n")
            f.write(f"  FN={cm[1][0]:4d}  TP={cm[1][1]:4d}\n\n")
            
            if r['feature_importance']:
                f.write("En Önemli 5 Etken:\n")
                sorted_fi = sorted(r['feature_importance'], key=lambda x: x[1], reverse=True)[:5]
                for i, (feat, imp) in enumerate(sorted_fi, 1):
                    f.write(f"  {i}. {feat}: {imp:.4f}\n")
            else:
                f.write("Feature Importance: Bu modelde çıkarılamaz\n")
            f.write("\n")
        
        # Özet
        best = sorted_results[0]
        best_recall = max(results, key=lambda x: x['recall'])
        
        f.write("=" * 70 + "\n")
        f.write("SONUÇ\n")
        f.write("=" * 70 + "\n")
        f.write(f"En İyi Model (ROC-AUC): {best['model']} ({best['roc_auc']:.4f})\n")
        f.write(f"En Yüksek Recall: {best_recall['model']} ({best_recall['recall']*100:.2f}%)\n")
        f.write("\nNOT: Sağlık verilerinde Recall kritiktir - felç vakalarını kaçırmamak önemli!\n")
    
    print(f"✅ Rapor kaydedildi: {report_path}")
    
    # 2. CSV Metrics
    csv_path = os.path.join(RESULTS_DIR, "metrics_comparison.csv")
    metrics_df = pd.DataFrame([{
        'Model': r['model'],
        'Filename': r['filename'],
        'Accuracy': r['accuracy'],
        'Precision': r['precision'],
        'Recall': r['recall'],
        'F1_Score': r['f1'],
        'ROC_AUC': r['roc_auc'],
        'Train_Time_Sec': r['train_time']
    } for r in results])
    metrics_df.to_csv(csv_path, index=False)
    print(f"✅ Metrikler kaydedildi: {csv_path}")
    
    # 3. Feature Importance CSV
    fi_path = os.path.join(RESULTS_DIR, "feature_importance.csv")
    fi_data = []
    for r in results:
        if r['feature_importance']:
            for feat, imp in r['feature_importance']:
                fi_data.append({'Model': r['model'], 'Feature': feat, 'Importance': imp})
    
    if fi_data:
        fi_df = pd.DataFrame(fi_data)
        fi_df.to_csv(fi_path, index=False)
        print(f"✅ Feature importance kaydedildi: {fi_path}")


def print_summary(results):
    """Özet tabloyu ekrana yazdırır."""
    print_header("SONUÇ ÖZETİ")
    
    sorted_results = sorted(results, key=lambda x: x['roc_auc'], reverse=True)
    
    print(f"{'Model':<25} {'Accuracy':>10} {'Recall':>10} {'AUC':>10}")
    print("-" * 55)
    
    medals = ["🥇", "🥈", "🥉", "  ", "  "]
    for i, r in enumerate(sorted_results):
        print(f"{medals[i]} {r['model']:<22} {r['accuracy']*100:>9.2f}% {r['recall']*100:>9.2f}% {r['roc_auc']:>10.4f}")
    
    print()
    best = sorted_results[0]
    print(f"🏆 En İyi Model: {best['model']} (AUC: {best['roc_auc']:.4f})")
    
    print("\n📁 Kaydedilen Dosyalar:")
    print("   saved_models/")
    for r in results:
        print(f"     ├── {r['filename']}")
    print("     ├── scaler.pkl")
    print("     └── feature_columns.pkl")
    print("   results/")
    print("     ├── training_report.txt")
    print("     ├── metrics_comparison.csv")
    print("     └── feature_importance.csv")


def main():
    """Ana fonksiyon."""
    print("\n" + "🏥 STROKE PREDICTION - MODEL EĞİTİM SİSTEMİ".center(60))
    print("=" * 60)
    
    start = time.time()
    
    # Veri hazırla
    X_train, X_test, y_train, y_test, feature_columns = load_and_prepare_data()
    
    # Modelleri eğit
    results = train_all_models(X_train, X_test, y_train, y_test, feature_columns)
    
    # Sonuçları kaydet
    write_results(results, feature_columns)
    
    # Özet
    print_summary(results)
    
    total_time = time.time() - start
    print(f"\n⏱️ Toplam süre: {total_time:.1f} saniye")
    print("\n✅ Tamamlandı! Artık 'python app_gradio.py' ile tahmin yapabilirsiniz.\n")


if __name__ == "__main__":
    main()
