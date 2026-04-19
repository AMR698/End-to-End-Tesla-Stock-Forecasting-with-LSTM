# 🚗 Tesla Stock Forecasting using LSTM

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

> تنبؤ بأسعار سهم Tesla (TSLA) باستخدام نموذج LSTM للتعلم العميق مع واجهة Streamlit تفاعلية.

---

## 📊 النتائج

| Metric | Value |
|--------|-------|
| RMSE   | **144.59** |
| MAPE   | **18.70%** |
| Accuracy | **~81.3%** |

---

## 📁 هيكل المشروع

```
Tesla-Stock-Forecasting/
│
├── app.py                  # Streamlit web app
├── requirements.txt        # المكتبات المطلوبة
├── .gitignore
├── README.md
└── Tesla_Stock_Forecasting_using_LSTM.ipynb  # Jupyter Notebook
```

---

## 🗂️ Dataset

- **Source:** [Kaggle — Tesla Stock Data](https://www.kaggle.com/datasets/varpit94/tesla-stock-data-updated-till-28jun2021)
- **Period:** 2010 → 2021
- **Rows:** 2,956 trading days
- **Target:** `Close` price

---

## 🧠 Model Architecture

```
Input (60 days) 
    → LSTM(100, return_sequences=True) 
    → Dropout(0.2)
    → LSTM(64, return_sequences=False) 
    → Dropout(0.2)
    → Dense(32, ReLU)
    → Dense(16, ReLU)
    → Dense(1)  ← predicted price
```

---

## 🚀 تشغيل المشروع

### 1. Clone الـ Repo
```bash
git clone https://github.com/YOUR_USERNAME/Tesla-Stock-Forecasting.git
cd Tesla-Stock-Forecasting
```

### 2. تثبيت المكتبات
```bash
pip install -r requirements.txt
```

### 3. تشغيل الـ Streamlit App
```bash
streamlit run app.py
```

### 4. ارفع ملف CSV
- حمّل الداتا من [Kaggle](https://www.kaggle.com/datasets/varpit94/tesla-stock-data-updated-till-28jun2021)
- ارفعه من الـ Sidebar في الـ app

---

## ⚙️ الـ App Features

- 📂 رفع ملف CSV مباشرة
- 📊 عرض البيانات مع charts تفاعلية
- ⚙️ تحكم كامل في إعدادات الموديل
- 🚀 تدريب مع progress bar في real-time
- 📉 مقارنة التنبؤ بالسعر الحقيقي
- 💾 تحميل الموديل والـ scaler بعد التدريب

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| UI | Streamlit |
| Deep Learning | TensorFlow / Keras |
| Data | Pandas / NumPy |
| Scaling | Scikit-learn |
| Visualization | Plotly |
| Model Saving | Keras + Joblib |

---

## 📦 Deploy على Streamlit Cloud

1. ارفع الكود على GitHub
2. روح على [share.streamlit.io](https://share.streamlit.io)
3. اربط الـ repo واختار `app.py`
4. اضغط Deploy ✅

---

## 📄 License

MIT License — feel free to use and modify.
