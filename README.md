# 🏥 Stroke Prediction (İnme Riski Tahmini)

Bu proje, makine öğrenmesi algoritmaları kullanarak bireylerin inme (felç) geçirme riskini tahmin etmeyi amaçlayan kapsamlı bir veri bilimi çalışmasıdır.

## 🚀 Özellikler

- **5 Farklı ML Modeli:** Logistic Regression, Random Forest, XGBoost, SVM ve KNN modellerinin karşılaştırmalı analizi.
- **Dengesiz Veri Yönetimi:** SMOTE (Synthetic Minority Over-sampling Technique) ile veri dengesizliğinin giderilmesi.
- **Kapsamlı Raporlama:** `main.py` çalıştığında detaylı eğitim raporları, metrik tabloları ve feature importance analizleri üretir.
- **İnteraktif Web Arayüzü:** Gradio tabanlı modern arayüz ile kullanıcı dostu tahmin imkanı.
- **Model Kayıt Sistemi:** Eğitilen modeller `.pkl` formatında kaydedilerek tekrar tekrar kullanılabilir.

## 📂 Proje Yapısı

```
stroke_prediction/
├── main.py              # Tüm modelleri eğiten ve raporlayan ana script
├── app_gradio.py        # Web tabanlı tahmin arayüzü
├── data_utils.py        # Veri işleme yardımcı fonksiyonları
├── healthcare-dataset-stroke-data.csv # Veri seti
├── saved_models/        # Eğitilmiş modellerin kaydedildiği klasör
├── results/             # Eğitim raporları ve metrik tabloları
└── models/              # (Opsiyonel) Tekli model scriptleri
```

## 🛠️ Kurulum

1. Projeyi klonlayın:
   ```bash
   git clone https://github.com/kadir-yilmaz/stroke_prediction.git
   cd stroke_prediction
   ```

2. Gerekli paketleri yükleyin:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost gradio joblib
   ```

## 💻 Kullanım

### 1. Modelleri Eğitme
Önce modelleri eğitmek ve sonuçları görmek için ana scripti çalıştırın:
```bash
python main.py
```
Bu işlem 5 modeli eğitecek, sonuçları `results/` klasörüne yazacak ve modelleri `saved_models/` klasörüne kaydedecektir.

### 2. Arayüzü Başlatma
Eğitim tamamlandıktan sonra web arayüzünü başlatın:
```bash
python app_gradio.py
```
Tarayıcınızda açılan arayüzden yaş, glikoz seviyesi, BMI gibi değerleri girerek risk tahmini yapabilirsiniz.

## 📊 Model Performansları

Proje kapsamında elde edilen örnek sonuçlar:

| Model | Accuracy | Recall | ROC-AUC |
|-------|----------|--------|---------|
| **Logistic Regression** | %74.95 | **%80.00** | **0.8445** |
| Random Forest | %83.07 | %48.00 | 0.7854 |
| XGBoost | %88.45 | %24.00 | 0.7821 |
| SVM (RBF) | %81.41 | %50.00 | 0.7794 |
| KNN (k=5) | %82.29 | %28.00 | 0.6853 |

> **Not:** Sağlık verilerinde **Recall** (Duyarlılık) kritiktir. Logistic Regression modeli %80 Recall ile felç vakalarını en iyi tespit eden model olmuştur.

## 🔍 Veri Seti
Kullanılan veri seti: [Healthcare Dataset Stroke Data](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- 5110 Gözlem
- 11 Öznitelik (Yaş, Cinsiyet, Hipertansiyon, Kalp Hastalığı...)
- Hedef Değişken: Stroke (0: Sağlıklı, 1: Felç)

