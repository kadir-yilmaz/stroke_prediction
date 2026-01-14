import os
import sys
import pandas as pd
import numpy as np
import gradio as gr
import joblib

# Proje dizinleri
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def check_models_exist():
    """Eğitilmiş modellerin varlığını kontrol eder."""
    required = ["scaler.pkl", "feature_columns.pkl"]
    for f in required:
        if not os.path.exists(os.path.join(SAVED_MODELS_DIR, f)):
            return False
    return True


def get_available_models():
    """Mevcut model dosyalarını listeler."""
    if not os.path.exists(SAVED_MODELS_DIR):
        return []
    
    models = []
    model_names = {
        "logistic_regression.pkl": "Logistic Regression",
        "random_forest.pkl": "Random Forest",
        "xgboost.pkl": "XGBoost",
        "svm_rbf.pkl": "SVM (RBF)",
        "knn_k5.pkl": "KNN (k=5)"
    }
    
    for filename in os.listdir(SAVED_MODELS_DIR):
        if filename.endswith(".pkl") and filename not in ["scaler.pkl", "feature_columns.pkl"]:
            display_name = model_names.get(filename, filename.replace(".pkl", "").replace("_", " ").title())
            models.append((display_name, filename))
    
    return models


def load_model(filename):
    """Model, scaler ve feature columns yükler."""
    model_path = os.path.join(SAVED_MODELS_DIR, filename)
    scaler_path = os.path.join(SAVED_MODELS_DIR, "scaler.pkl")
    columns_path = os.path.join(SAVED_MODELS_DIR, "feature_columns.pkl")
    
    if not all(os.path.exists(p) for p in [model_path, scaler_path, columns_path]):
        return None, None, None
    
    return (
        joblib.load(model_path),
        joblib.load(scaler_path),
        joblib.load(columns_path)
    )


def load_metrics():
    """Kaydedilmiş metrikleri yükler."""
    csv_path = os.path.join(RESULTS_DIR, "metrics_comparison.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


def format_metrics_table():
    """Metrik tablosunu formatlar."""
    df = load_metrics()
    if df is None:
        return "⚠️ Metrikler bulunamadı. Önce `python main.py` çalıştırın."
    
    # Sırala
    df = df.sort_values('ROC_AUC', ascending=False).reset_index(drop=True)
    
    output = "## 📊 Model Performans Karşılaştırması\n\n"
    output += "| # | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |\n"
    output += "|---|-------|----------|-----------|--------|----|---------|\n"
    
    medals = ["🥇", "🥈", "🥉"]
    for i, row in df.iterrows():
        medal = medals[i] if i < 3 else f"{i+1}."
        output += f"| {medal} | {row['Model']} | {row['Accuracy']*100:.1f}% | {row['Precision']*100:.1f}% | {row['Recall']*100:.1f}% | {row['F1_Score']*100:.1f}% | {row['ROC_AUC']:.4f} |\n"
    
    output += "\n> 💡 **İpucu:** Sağlık verilerinde **Recall** kritiktir - felç vakalarını kaçırmamak önemli!\n"
    
    return output


def predict_stroke(model_choice, age, gender, hypertension, heart_disease, ever_married,
                   work_type, residence_type, avg_glucose_level, bmi, smoking_status):
    """Seçilen model ile felç riskini tahmin eder."""
    
    # Model dosya adını bul
    model_filename = None
    for display_name, filename in get_available_models():
        if display_name == model_choice:
            model_filename = filename
            break
    
    if model_filename is None:
        return "⚠️ Model bulunamadı!", ""
    
    model, scaler, feature_columns = load_model(model_filename)
    
    if model is None:
        return "⚠️ Önce `python main.py` ile modelleri eğitin!", ""
    
    # Input hazırla
    input_data = {
        'age': float(age),
        'hypertension': 1 if hypertension == "Evet" else 0,
        'heart_disease': 1 if heart_disease == "Evet" else 0,
        'avg_glucose_level': float(avg_glucose_level),
        'bmi': float(bmi),
        'gender_Male': 1 if gender == "Male" else 0,
        'gender_Other': 1 if gender == "Other" else 0,
        'ever_married_Yes': 1 if ever_married == "Evet" else 0,
        'work_type_Never_worked': 1 if work_type == "Never_worked" else 0,
        'work_type_Private': 1 if work_type == "Private" else 0,
        'work_type_Self-employed': 1 if work_type == "Self-employed" else 0,
        'work_type_children': 1 if work_type == "children" else 0,
        'Residence_type_Urban': 1 if residence_type == "Urban" else 0,
        'smoking_status_formerly smoked': 1 if smoking_status == "formerly smoked" else 0,
        'smoking_status_never smoked': 1 if smoking_status == "never smoked" else 0,
        'smoking_status_smokes': 1 if smoking_status == "smokes" else 0,
    }
    
    # DataFrame oluştur
    df_input = pd.DataFrame([input_data])
    for col in feature_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[feature_columns]
    
    # Scale ve predict
    X_scaled = scaler.transform(df_input)
    prediction = model.predict(X_scaled)[0]
    
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(X_scaled)[0]
        risk_score = probability[1] * 100
    else:
        risk_score = prediction * 100
        probability = [1 - prediction, prediction]
    
    # Sonuç formatla
    if prediction == 1:
        result = f"⚠️ YÜKSEK RİSK! Felç riski: %{risk_score:.1f}"
    else:
        result = f"✅ Düşük risk. Felç riski: %{risk_score:.1f}"
    
    details = f"""
### 🤖 Model: {model_choice}

| Sınıf | Olasılık |
|-------|----------|
| Sağlıklı (0) | %{probability[0]*100:.1f} |
| Felç (1) | %{probability[1]*100:.1f} |

**Risk Skoru:** %{risk_score:.1f}

---

#### Girilen Veriler:
- Yaş: {age}
- Cinsiyet: {gender}
- Hipertansiyon: {hypertension}
- Kalp Hastalığı: {heart_disease}
- Glikoz: {avg_glucose_level} mg/dL
- BMI: {bmi}
"""
    
    return result, details


def get_example_data():
    """CSV'den örnek veriler alır."""
    filepath = os.path.join(PROJECT_ROOT, "healthcare-dataset-stroke-data.csv")
    if not os.path.exists(filepath):
        return []
    
    df = pd.read_csv(filepath)
    
    examples = []
    for _, row in df.head(8).iterrows():
        bmi_val = row['bmi'] if pd.notna(row['bmi']) else 28.0
        try:
            bmi_val = float(bmi_val)
        except:
            bmi_val = 28.0
        
        examples.append([
            row['age'],
            row['gender'],
            "Evet" if row['hypertension'] == 1 else "Hayır",
            "Evet" if row['heart_disease'] == 1 else "Hayır",
            "Evet" if row['ever_married'] == "Yes" else "Hayır",
            row['work_type'],
            row['Residence_type'],
            row['avg_glucose_level'],
            bmi_val,
            row['smoking_status']
        ])
    
    return examples


def create_interface():
    """Gradio arayüzünü oluşturur."""
    
    # Mevcut modelleri al
    available_models = get_available_models()
    model_names = [name for name, _ in available_models] if available_models else ["Model bulunamadı"]
    
    examples = get_example_data()
    
    with gr.Blocks(
        title="🏥 Stroke Prediction"
    ) as demo:
        
        gr.Markdown("""
        # 🏥 İnme (Felç) Risk Tahmini
        
        Eğitilmiş ML modelleri ile felç riski tahmini yapın.
        
        > ⚠️ **Not:** Modeller eğitilmemişse önce `python main.py` çalıştırın!
        """)
        
        with gr.Tabs():
            # TAB 1: Risk Tahmini
            with gr.Tab("🔍 Risk Tahmini"):
                gr.Markdown("### Kişisel Verilerle Felç Riski Tahmini")
                
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=model_names,
                        value=model_names[0] if model_names else None,
                        label="🤖 Model Seçin",
                        info="Tahmin için kullanılacak model"
                    )
                
                with gr.Row():
                    with gr.Column():
                        age = gr.Slider(0, 100, value=50, label="Yaş", step=1)
                        gender = gr.Dropdown(["Male", "Female", "Other"], value="Male", label="Cinsiyet")
                        ever_married = gr.Radio(["Evet", "Hayır"], value="Evet", label="Evli mi?")
                        work_type = gr.Dropdown(
                            ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
                            value="Private", label="Çalışma Tipi"
                        )
                        residence_type = gr.Dropdown(["Urban", "Rural"], value="Urban", label="Yerleşim")
                    
                    with gr.Column():
                        hypertension = gr.Radio(["Evet", "Hayır"], value="Hayır", label="Hipertansiyon")
                        heart_disease = gr.Radio(["Evet", "Hayır"], value="Hayır", label="Kalp Hastalığı")
                        avg_glucose_level = gr.Slider(50, 300, value=100, label="Glikoz (mg/dL)", step=1)
                        bmi = gr.Slider(10, 60, value=25, label="BMI", step=0.1)
                        smoking_status = gr.Dropdown(
                            ["never smoked", "formerly smoked", "smokes", "Unknown"],
                            value="never smoked", label="Sigara"
                        )
                
                predict_btn = gr.Button("🔍 Risk Analizi Yap", variant="primary", size="lg")
                
                with gr.Row():
                    result_text = gr.Textbox(label="Sonuç", lines=2)
                    result_details = gr.Markdown()
                
                predict_btn.click(
                    fn=predict_stroke,
                    inputs=[model_dropdown, age, gender, hypertension, heart_disease, ever_married,
                           work_type, residence_type, avg_glucose_level, bmi, smoking_status],
                    outputs=[result_text, result_details]
                )
                
                if examples:
                    gr.Markdown("### 📋 Örnek Veriler (CSV'den)")
                    gr.Examples(
                        examples=examples,
                        inputs=[age, gender, hypertension, heart_disease, ever_married,
                               work_type, residence_type, avg_glucose_level, bmi, smoking_status]
                    )
            
            # TAB 2: Model Karşılaştırma
            with gr.Tab("📊 Model Karşılaştırma"):
                gr.Markdown("### Eğitilmiş Modellerin Performansı")
                
                metrics_output = gr.Markdown(format_metrics_table())
                
                refresh_btn = gr.Button("🔄 Yenile", variant="secondary")
                refresh_btn.click(fn=format_metrics_table, inputs=[], outputs=[metrics_output])
                
                gr.Markdown("""
                ---
                ### 📁 Kaydedilen Dosyalar
                
                ```
                saved_models/
                ├── logistic_regression.pkl
                ├── random_forest.pkl
                ├── xgboost.pkl
                ├── svm_rbf.pkl
                ├── knn_k5.pkl
                ├── scaler.pkl
                └── feature_columns.pkl
                
                results/
                ├── training_report.txt
                ├── metrics_comparison.csv
                └── feature_importance.csv
                ```
                """)
    
    return demo


if __name__ == "__main__":
    print("🚀 Gradio uygulaması başlatılıyor...")
    
    if not check_models_exist():
        print("⚠️ Eğitilmiş model bulunamadı!")
        print("   Önce şu komutu çalıştırın: python main.py")
        print()
    
    print(f"📁 Modeller: {SAVED_MODELS_DIR}")
    demo = create_interface()
    demo.launch(share=False)
