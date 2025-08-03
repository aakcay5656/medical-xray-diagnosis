
# 🧠 Medikal Görüntü Analizi ve Danışman Sistemi

Bu proje, akciğer röntgenleri ve beyin MR görüntülerini analiz eden yapay zeka destekli bir medikal danışmanlık sistemidir. Kullanıcılar, medikal görüntüler yükleyerek bu görüntüler hakkında doğal dilde sorular sorabilir ve modelden anlamlı, tıbbi açıklamalar alabilir.

## ✨ Özellikler

* 📷 **Görüntü Analizi:** Akciğer röntgeni ve beyin MR görüntülerini sınıflandırır.
* 🧠 **LLM Destekli Soru-Cevap:** Kullanıcı sorularına Gemini ile yanıt verir.
* 🗂️ **Chat Geçmişi:** Önceki konuşmalarla bağlamı korur.
* 🧩 **Modüler Yapı:** Yeni organ modelleri kolayca entegre edilebilir.
* 🌐 **Web Arayüzü:** Streamlit tabanlı sade kullanıcı arayüzü.

## 🏗️ Kullanılan Teknolojiler

* Python
* TensorFlow / PyTorch (CNN modelleri için)
* Gemini (Google AI - LLM destekli yanıtlar için)
* LangChain (LLM orkestrasyonu)
* Streamlit (Frontend arayüz)
  
## 🚀 Kurulum

### 1. Depoyu klonlayın

```bash
git clone https://github.com/aakcay5656/medical-xray-diagnosis.git
cd medikal-analiz-ai
```

### 2. Ortamı oluşturun

```bash
python -m venv .venv
source .venv/bin/activate  # Windows için: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Ortam değişkenlerini tanımlayın

`.env` dosyasını oluşturun ve Gemini API anahtarınızı girin:

```
GOOGLE_API_KEY=your_key_here
```

### 4. Uygulamayı başlatın

```bash
uvicorn app.api.main:app --reload

streamlit run app/app.py
```



