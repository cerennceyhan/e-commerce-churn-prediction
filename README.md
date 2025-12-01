# E-Commerce Product Churn Risk Prediction 🚀

This project is an end-to-end data science pipeline designed to detect "Quality Churn" and "Engagement Churn" risks in e-commerce products using **LLM (Claude 4.5 Sonnet)** and **XGBoost**.

## ⚠️ Important Privacy & Data Policy
Due to **GDPR/KVKK** regulations and strict data privacy protocols:

* **Synthetic Data Only:** The dataset included in this repository (`data/raw/sample_dataset.csv`) is **synthetic/fake**. It is generated to demonstrate the code's functionality and structure without violating privacy.
* **Real Data Privacy:** The original dataset contains real customer reviews, names, and sensitive information. Therefore, raw real data and intermediate processing files (`base_metrics.csv`, `llm_extraction.csv`) are **excluded** from this repository.
* **Proven Results:** The visualization outputs (SHAP plots, Confusion Matrices) located in the `outputs/` folder are generated using the **REAL dataset** to showcase the actual performance and validity of the model.

## 📂 Project Structure

* `ty_scrapping.py`: Selenium-based web scraper customized for product reviews.
* `base_metrics.py`: Feature engineering module that calculates independent metrics (rating deviation, review velocity, etc.).
* `llm_extraction.py`: Advanced feature extraction using **Anthropic Claude 4.5 Sonnet API**. It analyzes unstructured text to detect specific issues like fitment problems, fabric quality, and color mismatches.
* `train_model.py`: Trains an XGBoost classifier with SMOTE oversampling and generates SHAP explanations for interpretability.
* `data/`: Contains the synthetic raw dataset (`sample_dataset.csv`) for testing.
* `outputs/`: Contains performance graphs based on real-world data analysis.

## 📊 Model Performance (on Real Data)
The model successfully differentiates between *Healthy*, *Quality Churn*, and *Engagement Churn* products using hybrid features.

*(See `outputs/` folder for detailed SHAP analysis and Confusion Matrix)*

## 🛠️ How to Run
1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Environment Setup:**
    Create a `.env` file in the root directory and add your API key:
    ```text
    ANTHROPIC_API_KEY=sk-your-api-key-here
    ```
3.  **Execution Pipeline:**
    ```bash
    python base_metrics.py    # Step 1: Base stats
    python llm_extraction.py  # Step 2: LLM Analysis (Claude 4.5)
    python train_model.py     # Step 3: Training & Evaluation
    ```

---

# E-Ticaret Ürün Churn (Risk) Analizi ve Tahmini 🇹🇷

Bu proje, e-ticaret ürünlerindeki "Kalite Kaynaklı Müşteri Kaybı" (Quality Churn) ve "İlgi Kaybı" (Engagement Churn) risklerini tespit etmek için geliştirilmiş uçtan uca bir veri bilimi projesidir. Projede **LLM (Claude 4.5 Sonnet)** ve **XGBoost** algoritmaları hibrit olarak kullanılmıştır.

## ⚠️ Önemli: Veri Gizliliği ve KVKK Politikası
Bu projeyi incelerken lütfen aşağıdaki veri gizliliği kurallarını göz önünde bulundurun:

1.  **Sentetik (Fake) Veri:** Bu depoda yer alan `data/raw/sample_dataset.csv` dosyası, kodların çalışırlığını test edebilmeniz için oluşturulmuş **tamamen sahte/sentetik** verilerdir. Gerçek kişi veya kurumlarla ilgisi yoktur.
2.  **Gerçek Veriler:** Projenin geliştirilmesinde kullanılan, gerçek müşteri isimleri ve yorumlarını içeren ham veri seti ve ara işlem dosyaları (`base_metrics.csv` vb.), **KVKK (Kişisel Verilerin Korunması Kanunu)** ve gizlilik esasları gereği bu depoda **paylaşılmamıştır**.
3.  **Kanıtlanmış Sonuçlar:** `outputs/` klasöründe göreceğiniz grafikler (SHAP analizi, Confusion Matrix), modelin **GERÇEK verilerle** eğitilmesi sonucu elde edilen başarıyı göstermektedir.

## 📂 Proje Dosya Yapısı

* `ty_scrapping.py`: Trendyol ürün yorumlarını çekmek için geliştirilmiş, Selenium tabanlı web kazıma botu.
* `base_metrics.py`: Ürünler için sayısal özellikleri (puan ortalaması, yorum sıklığı, standart sapma vb.) hesaplayan modül.
* `llm_extraction.py`: **Anthropic Claude 4.5 Sonnet API** kullanarak yorum metinlerini analiz eden yapay zeka modülü. Metinlerden "kalıp hatası", "kumaş kalitesi", "renk uyuşmazlığı" gibi spesifik sorunları tespit eder.
* `train_model.py`: Elde edilen tüm özellikleri birleştirerek XGBoost modeli ile risk tahmini yapar. SMOTE ile veri dengesizliğini giderir ve SHAP kütüphanesi ile modelin kararlarını açıklar.
* `outputs/`: Modelin gerçek veriler üzerindeki performans grafiklerini içerir.

## 📊 Model Performansı
Model, sayısal veriler ve LLM'den gelen içgörüleri birleştirerek ürünleri 3 sınıfa ayırmaktadır:
1.  **Healthy (Sağlıklı):** Sorunsuz ürünler.
2.  **Quality Churn Risk:** Kalite, kalıp veya kumaş sorunu olan iade riski yüksek ürünler.
3.  **Engagement Churn Risk:** Yorum sayısı az veya fiyat/performans dengesi bozuk ürünler.

*(Detaylı performans grafikleri için `outputs` klasörüne bakabilirsiniz.)*

## 🛠️ Kurulum ve Çalıştırma

1.  **Gerekli Kütüphaneler:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **API Ayarları:**
    Ana dizinde `.env` dosyası oluşturun ve Anthropic API anahtarınızı ekleyin:
    ```text
    ANTHROPIC_API_KEY=sk-ant-api03-...
    ```

3.  **Çalıştırma Sırası:**
    ```bash
    python base_metrics.py    # Adım 1: Temel metrikleri çıkar
    python llm_extraction.py  # Adım 2: Yapay zeka (Claude 4.5) ile yorumları analiz et
    python train_model.py     # Adım 3: Modeli eğit ve sonuçları üret
    ```

## 📝 Hazırlayan / Author
**Ceren Ceyhan**
