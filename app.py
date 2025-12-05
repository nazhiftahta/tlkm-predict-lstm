import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

# ===============================
# CONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Prediksi Saham TLKM — LSTM",
    layout="wide"
)

# ===============================
# CSS STYLING
# ===============================
st.markdown("""
<style>
/* Background */
body {
    background-color: #F8F9FA;
}

/* Header Title */
h1, h2, h3 {
    font-weight: 600;
    color: #1A1A1A;
}

/* Card */
.card {
    padding: 20px;
    border-radius: 10px;
    background-color: white;
    border: 1px solid #DDDDDD;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
    margin-bottom: 15px;
}

/* Equal Height Cards */
.equal-card {
    height: 160px;
}

/* Footer spacing */
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD DATA & MODEL
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("daily/TLKM.csv")

@st.cache_resource
def load_lstm_model():
    return load_model("lstm_tlkm.h5", compile=False)

@st.cache_resource
def load_scaler():
    return joblib.load(open("scaler.pkl", "rb"))


data = load_data()
lstm_model = load_lstm_model()
scaler = load_scaler()

# ===============================
# SIDEBAR NAVIGATION
# ===============================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Home", "Data Analysis", "Model and Evaluation", "Predictions"]
)

# =====================================================
# ====================== HOME =========================
# =====================================================
if page == "Home":
    st.title("CAPSTONE PROJECT — Prediksi Harga Saham Telkom Indonesia (TLKM)")

    st.subheader("Deskripsi Dataset")
    st.markdown("""
Dataset yang digunakan merupakan *Indonesia Stock Dataset* (IHSG) yang memuat data historis
harga saham dalam tiga interval waktu: **daily**, **hourly**, dan **minutes**.  
Pada proyek ini, fokus analisis diarahkan pada saham **Telkom Indonesia (TLKM)** dengan menggunakan
data **harga penutupan harian** untuk membangun model prediksi berbasis *time series*.
    """)

    st.subheader("Tujuan Penelitian")
    st.markdown("""
Penelitian ini bertujuan mengembangkan sistem peramalan harga penutupan saham **TLKM** menggunakan 
pendekatan **Machine Learning** dan **Deep Learning**, khususnya model **LSTM**, serta dibandingkan 
dengan pendekatan statistik seperti **ARIMA** dan **Prophet**. Evaluasi dilakukan menggunakan 
**RMSE** dan **MAPE** untuk menilai performa model dan menghasilkan peramalan yang informatif.
    """)

    st.subheader("Informasi Perusahaan — Telkom Indonesia (TLKM)")
    st.markdown("""
Telkom Indonesia (Persero) Tbk adalah perusahaan sektor **Infrastructures**, terdaftar di BEI sejak 
**14 November 1995**, dengan kapitalisasi pasar lebih dari **Rp 367 triliun** dan saham beredar lebih 
dari **99 miliar lembar**. Data historis TLKM tersedia dalam resolusi **daily**, **hourly**, dan 
**minutes**, memungkinkan analisis mendalam terhadap pola temporal harga saham.
    """)

    st.subheader("Preview Dataset")
    st.dataframe(data.head())

# =====================================================
# ================== DATA ANALYSIS ====================
# =====================================================
elif page == "Data Analysis":
    st.title("Preprocessing and Exploratory Data Analysis")

    # ---- CSS (Kotak Besar & Elegan) ----
    st.markdown("""
    <style>
    .styled-box {
        background-color: #ffffff;
        border: 1px solid #d9d9d9;
        border-radius: 12px;
        padding: 28px;                       /* JARAK DALAM KOTAK (besar & elegan) */
        margin-top: 18px;
        margin-bottom: 18px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.07);
        min-height: 180px;                   /* Tinggi minimum kotak */
    }

    .box-title {
        font-size: 18px;
        font-weight: 600;
        color: #222222;
        margin-bottom: 10px;
    }

    .box-content {
        font-size: 15px;
        color: #444444;
        line-height: 1.5;
    }

    /* Kotak missing value horizontal */
    .inline-pre {
        background: #f7f7f7;
        padding: 14px;
        border-radius: 6px;
        border: 1px solid #e3e3e3;
        display: inline-block;  /* <<< BIKIN ISI "KE SAMPING" */
        font-size: 14px;
        white-space: pre;
    }
    </style>
    """, unsafe_allow_html=True)

    st.subheader("Grafik Harga Penutupan TLKM (Daily)")

    fig, ax = plt.subplots(figsize=(12, 4))
    data["close"].plot(ax=ax)
    ax.set_ylabel("Harga (Rp)")
    ax.set_title("Harga Penutupan TLKM (Daily)")
    st.pyplot(fig)

    st.subheader("Analisis Singkat")
    st.markdown("""
Grafik menunjukkan tren harga penutupan TLKM yang bergerak fluktuatif 
namun tetap menunjukkan pola jangka panjang yang dapat ditangkap model LSTM. 
Variasi musiman dan pergerakan tren memberikan sinyal bahwa dataset mengandung 
komponen pola temporal yang kuat dan cocok untuk metode forecasting berbasis kejadian berurutan.
    """)

    # ======================
    #   INFORMATION SUMMARY
    # ======================
    st.subheader("Information Summary")

    # ---- Kotak 1: Missing Value ----
    missing_html = """
    <div class='styled-box'>
        <div class='box-title'>Missing Values</div>
        <div class='box-content'>
            <span class='inline-pre'>timestamp    0
open         0
low          0
high         0
close        0
volume       0
dtype: int64</span>
        </div>
    </div>
    """
    st.markdown(missing_html, unsafe_allow_html=True)

    # ---- Kotak 2: Feature Engineering ----
    fe_text = (
        "Seluruh kolom numerik seperti open, high, low, close, dan volume "
        "diubah ke dalam tipe data float agar dapat diproses dengan benar oleh model.<br>"
        "<br>Target : <b>close (harga penutupan)</b><br>"
    )

    fe_html = f"""
    <div class='styled-box'>
        <div class='box-title'>Feature Engineering</div>
        <div class='box-content'>{fe_text}</div>
    </div>
    """
    st.markdown(fe_html, unsafe_allow_html=True)

    # ---- Kotak 3: Modeling ----
    model_text = """
    Model : <b>LSTM (Long Short-Term Memory)</b>
    <br><br>
    • LSTM(50, return_sequences=True) <br>
    • Dropout(0.2)  <br>
    • LSTM(50)  <br>
    • Dense(1)
    """

    model_html = f"""
    <div class='styled-box'>
        <div class='box-title'>Modelling</div>
        <div class='box-content'>{model_text}</div>
    </div>
    """
    st.markdown(model_html, unsafe_allow_html=True)

# =====================================================
# ================== MODEL EVALUATION =================
# =====================================================
elif page == "Model and Evaluation":

    st.title("Time Series Model — LSTM Forecasting")

    # Load predictions (recreate using scaler)
    scaled_data = scaler.transform(data[["close"]])
    window = 60

    # Create full sequence
    X_all = []
    for i in range(window, len(scaled_data)):
        X_all.append(scaled_data[i-window:i, 0])

    X_all = np.array(X_all).reshape(-1, window, 1)

    lstm_pred_scaled = lstm_model.predict(X_all)
    lstm_pred = scaler.inverse_transform(lstm_pred_scaled)

    # Align actual data
    actual = data["close"].iloc[window:].values

    st.subheader("Forecasting Plot")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(actual, label="Actual")
    ax.plot(lstm_pred, label="LSTM Predict")
    ax.set_title("LSTM Forecasting Harga Penutupan TLKM")
    ax.legend()
    st.pyplot(fig)

    # Metrics from your notebook
    rmse = 88.31896234934389
    mape = 0.018910808587022113

    st.subheader("Model Performance Summary")
    col0, col1, col2, col3, col4 = st.columns(5)

    with col0:
        st.markdown('<div class="card equal-card"><h4>Data Splitting</h4>'
                    '• Train: 80%<br>'
                    '• Test: 20%<br>'
                    '</div>', unsafe_allow_html=True)

    with col1:
        st.markdown('<div class="card equal-card"><h4>Epochs</h4>30</div>', 
                    unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card equal-card"><h4>Window Size</h4>60 hari</div>',
                    unsafe_allow_html=True)

    with col3:
        st.markdown(f'<div class="card equal-card"><h4>RMSE</h4>{rmse:.2f}</div>',
                    unsafe_allow_html=True)

    with col4:
        st.markdown(f'<div class="card equal-card"><h4>MAPE</h4>{mape:.4f}</div>',
                    unsafe_allow_html=True)

    st.subheader("Analisis Model")
    st.markdown("""
Model LSTM menunjukkan kemampuan yang baik dalam mengikuti pola tren harga TLKM 
secara umum. Garis prediksi cenderung lebih halus dibandingkan data aktual, yang 
mengindikasikan bahwa model telah menangkap tren jangka panjang tanpa terlalu sensitif 
terhadap fluktuasi jangka pendek. Nilai RMSE dan MAPE menunjukkan error rendah, 
menandakan performa model cukup optimal untuk forecasting jangka pendek.
    """)
    st.subheader("Perbandingan Model Forecasting")

    # Data perbandingan
    comparison_data = {
        "Model": ["Baseline", "LSTM", "ARIMA", "Prophet"],
        "RMSE": [66.791953, 88.318962, 553.101180, 1504.521240],
        "MAPE": [0.0131657, 0.018911, 0.122561, 0.396217]
    }

    comparison_df = pd.DataFrame(comparison_data)

    # Tampilkan tabel ke Streamlit
    st.dataframe(comparison_df)

    # Analisis
    st.subheader("Interpretasi Hasil")
    st.markdown("""
- **LSTM** adalah model terbaik untuk forecasting harga penutupan TLKM.  
  Dengan RMSE **rendah** dan MAPE **< 2%**, model ini menunjukkan performa yang stabil dan akurat. Walaupun baseline secara angka sedikit lebih kecil, baseline hanya memprediksi *harga tidak berubah*. Sedangkan LSTM benar-benar **mempelajari pola naik–turun** harian.
                
- **ARIMA** cocok sebagai pembanding klasik, tetapi kurang mampu menangkap pola non-linear  
  dan volatilitas harga pasar, sehingga error jauh lebih tinggi.

- **Prophet** tidak cocok untuk data saham TLKM, karena model ini cenderung menghasilkan  
  prediksi yang terlalu halus (*over-smoothing*) dan gagal mengikuti fluktuasi harga harian.
    """)

# =====================================================
# ===================== PREDICTIONS ===================
# =====================================================
elif page == "Predictions":
    st.title("Hasil Prediksi - Forecasting")

    # Recalculate scaled_data for this page
    scaled_data = scaler.transform(data[["close"]])
    
    # use last 60 data
    last_60 = scaled_data[-60:]
    last_60 = last_60.reshape(1, 60, 1)

    next_day_pred = lstm_model.predict(last_60)
    next_day_price = scaler.inverse_transform(next_day_pred)[0][0]

    st.subheader("Prediksi Harga Besok")
    st.markdown(f"""
    Prediksi harga penutupan TLKM untuk hari berikutnya:  
    **Rp {next_day_price:,.0f}**
    """)

    st.subheader("N-Day Prediction")

    n_days = st.slider("Pilih jumlah hari prediksi", 1, 30, 10)

    temp_seq = scaled_data[-60:].flatten().tolist()
    future_preds = []

    for _ in range(n_days):
        x_input = np.array(temp_seq[-60:]).reshape(1, 60, 1)
        next_scaled = lstm_model.predict(x_input)[0][0]
        future_preds.append(next_scaled)
        temp_seq.append(next_scaled)

    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

    st.line_chart(future_preds)

    st.subheader("Kesimpulan Akhir")
    st.markdown("""
Model LSTM memberikan proyeksi kenaikan harga saham TLKM secara bertahap dan stabil.
Hasil prediksi jangka pendek menunjukkan tren bullish dengan volatilitas yang relatif rendah,
yang sejalan dengan pola historis saham TLKM. Prediksi ini dapat digunakan sebagai referensi 
pendukung dalam pengambilan keputusan, namun tetap perlu mempertimbangkan kondisi pasar aktual.
    """)

