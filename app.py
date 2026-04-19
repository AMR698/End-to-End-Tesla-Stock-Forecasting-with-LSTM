import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tesla Stock Forecasting",
    page_icon="🚗",
    layout="wide"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2d3250);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #e50914;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #e50914; }
    .metric-label { font-size: 0.9rem; color: #aaa; margin-top: 4px; }
    .title-text {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #e50914, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stButton>button {
        background: linear-gradient(90deg, #e50914, #cc0000);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 30px;
        font-size: 1rem;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover { background: linear-gradient(90deg, #ff1a1a, #e50914); }
</style>
""", unsafe_allow_html=True)

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="title-text">🚗 Tesla Stock Forecasting using LSTM</p>', unsafe_allow_html=True)
st.markdown("**تنبؤ بأسعار سهم Tesla باستخدام نموذج LSTM**")
st.divider()

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Tesla_Motors.svg/320px-Tesla_Motors.svg.png", width=150)
    st.markdown("## ⚙️ الإعدادات")
    
    uploaded_file = st.file_uploader("📂 ارفع ملف CSV للبيانات", type=["csv"])
    
    st.markdown("---")
    st.markdown("### 🧠 إعدادات الموديل")
    look_back     = st.slider("Look Back (أيام)", 30, 120, 60, 10)
    train_split   = st.slider("نسبة بيانات التدريب", 0.70, 0.90, 0.80, 0.05)
    epochs        = st.slider("عدد الـ Epochs", 50, 200, 100, 10)
    batch_size    = st.selectbox("Batch Size", [16, 32, 64], index=1)
    
    st.markdown("---")
    st.markdown("### 📌 معلومات")
    st.info("**RMSE:** 144.59\n\n**MAPE:** 18.70%\n\n**Dataset:** 2956 يوم تداول\n\n**الفترة:** 2010 → 2021")

# ─── Helper Functions ──────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mean_absolute_error'])
    return model

def create_sequences(data, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# ─── Main Logic ────────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.warning("👈 ارفع ملف CSV من الـ Sidebar عشان تبدأ — الملف المطلوب هو `TSLA.csv` من Kaggle")
    
    st.markdown("### 📋 الشكل المطلوب للـ CSV:")
    sample = pd.DataFrame({
        'Date':      ['2010-06-29', '2010-06-30', '2010-07-01'],
        'Open':      [3.80, 5.00, 5.85],
        'High':      [5.00, 6.08, 6.15],
        'Low':       [3.51, 4.66, 5.60],
        'Close':     [4.78, 4.77, 5.74],
        'Adj Close': [4.78, 4.77, 5.74],
        'Volume':    [8218800, 6866900, 6091900]
    })
    st.dataframe(sample, use_container_width=True)
    st.stop()

# ─── Load & Show Data ──────────────────────────────────────────────────────────
data = load_data(uploaded_file)

st.markdown("## 📊 استكشاف البيانات")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{len(data):,}</div>
        <div class="metric-label">عدد الأيام</div></div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">${data['Close'].max():.0f}</div>
        <div class="metric-label">أعلى سعر إغلاق</div></div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">${data['Close'].min():.2f}</div>
        <div class="metric-label">أدنى سعر إغلاق</div></div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">${data['Close'].iloc[-1]:.2f}</div>
        <div class="metric-label">آخر سعر إغلاق</div></div>""", unsafe_allow_html=True)

st.markdown("---")

# ─── Raw Data Table ────────────────────────────────────────────────────────────
with st.expander("📋 عرض البيانات الخام"):
    st.dataframe(data.tail(20), use_container_width=True)

# ─── Price Chart ───────────────────────────────────────────────────────────────
st.markdown("### 📈 سعر سهم Tesla عبر الزمن")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=data['Date'], y=data['Open'],  name='Open',  line=dict(color='purple', width=1)))
fig1.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close', line=dict(color='#e50914', width=1.5)))
fig1.update_layout(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=400,
    legend=dict(orientation='h', yanchor='bottom', y=1.02),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor='#2d3250', title='السعر (USD)')
)
st.plotly_chart(fig1, use_container_width=True)

# ─── Volume Chart ──────────────────────────────────────────────────────────────
with st.expander("📊 حجم التداول"):
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(x=data['Date'], y=data['Volume'], name='Volume', marker_color='#e50914', opacity=0.7))
    fig_vol.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300)
    st.plotly_chart(fig_vol, use_container_width=True)

st.divider()

# ─── Train Model ───────────────────────────────────────────────────────────────
st.markdown("## 🧠 تدريب الموديل")

if st.button("🚀 ابدأ التدريب والتنبؤ"):

    with st.spinner("جاري تجهيز البيانات..."):
        dataset    = data[['Close']].values
        scaler     = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(dataset)

        train_size = int(len(data_scaled) * train_split)
        train_data = data_scaled[:train_size, 0:1]
        test_data  = data_scaled[train_size - look_back:, 0:1]

        X_train, y_train = create_sequences(train_data, look_back)
        X_test,  y_test  = create_sequences(test_data,  look_back)

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test  = X_test.reshape((X_test.shape[0],  X_test.shape[1],  1))

    st.success(f"✅ البيانات جاهزة — Training: {len(X_train):,} | Testing: {len(X_test):,}")

    with st.spinner("جاري بناء وتدريب الموديل..."):
        model = build_model((look_back, 1))

        progress_bar = st.progress(0, text="جاري التدريب...")
        loss_chart_placeholder = st.empty()
        loss_history = []

        class StreamlitCallback(EarlyStopping):
            def on_epoch_end(self, epoch, logs=None):
                super().on_epoch_end(epoch, logs)
                loss_history.append(logs.get('loss', 0))
                pct = min(int((epoch + 1) / epochs * 100), 100)
                progress_bar.progress(pct, text=f"Epoch {epoch+1}/{epochs} — Loss: {logs.get('loss', 0):.6f}")
                if len(loss_history) > 1:
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(y=loss_history, mode='lines', line=dict(color='#e50914')))
                    fig_loss.update_layout(
                        title="Training Loss", template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        height=250, margin=dict(t=40, b=20),
                        xaxis=dict(title='Epoch'), yaxis=dict(title='Loss')
                    )
                    loss_chart_placeholder.plotly_chart(fig_loss, use_container_width=True)

        cb = StreamlitCallback(monitor='loss', patience=15, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[cb], verbose=0)
        progress_bar.progress(100, text="✅ اكتمل التدريب!")

    with st.spinner("جاري التنبؤ..."):
        predictions = model.predict(X_test, verbose=0)
        predictions = scaler.inverse_transform(predictions)
        y_test_inv  = scaler.inverse_transform(y_test.reshape(-1, 1))

        rmse = np.sqrt(mean_squared_error(y_test_inv, predictions))
        mape = np.mean(np.abs((y_test_inv - predictions) / y_test_inv)) * 100

    # ─── Results Metrics ───────────────────────────────────────────────────────
    st.markdown("## 📊 نتائج الموديل")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{rmse:.2f}</div>
            <div class="metric-label">RMSE</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{mape:.2f}%</div>
            <div class="metric-label">MAPE</div></div>""", unsafe_allow_html=True)
    with c3:
        accuracy = max(0, 100 - mape)
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{accuracy:.1f}%</div>
            <div class="metric-label">Accuracy</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ─── Prediction Chart ──────────────────────────────────────────────────────
    st.markdown("### 📉 السعر الحقيقي vs التنبؤ")
    test_dates = data['Date'].iloc[train_size:].reset_index(drop=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=test_dates, y=y_test_inv.flatten(), name='السعر الحقيقي',  line=dict(color='#00b4d8', width=2)))
    fig2.add_trace(go.Scatter(x=test_dates, y=predictions.flatten(),  name='التنبؤ',        line=dict(color='#e50914', width=2, dash='dash')))
    fig2.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=450,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        xaxis=dict(showgrid=False, title='التاريخ'),
        yaxis=dict(showgrid=True, gridcolor='#2d3250', title='السعر (USD)')
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ─── Save Model ────────────────────────────────────────────────────────────
    st.markdown("### 💾 حفظ الموديل")
    model.save("stock_model.keras")
    joblib.dump(scaler, "scaler.pkl")

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        with open("stock_model.keras", "rb") as f:
            st.download_button("⬇️ تحميل الموديل (.keras)", f, "stock_model.keras", use_container_width=True)
    with col_dl2:
        with open("scaler.pkl", "rb") as f:
            st.download_button("⬇️ تحميل الـ Scaler (.pkl)", f, "scaler.pkl", use_container_width=True)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("<center style='color:#555'>Tesla Stock Forecasting — LSTM Model | Built with Streamlit 🚀</center>", unsafe_allow_html=True)
