import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from datetime import timedelta
import tensorflow as tf

model = load_model(
    "lstm_tlkm.h5",
    custom_objects={"mse": tf.keras.losses.MeanSquaredError()}
)
scaler = joblib.load("scaler.pkl")

st.title("📈 Prediksi Harga Penutupan Saham TLKM (LSTM)")

uploaded = st.file_uploader("Upload file CSV harga historis TLKM", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    
    st.write("Data Historis:")
    st.dataframe(df.tail())

    data = df["Close"].values.reshape(-1, 1)
    scaled_data = scaler.transform(data)

    # ambil window 60 terakhir
    window = scaled_data[-60:]
    X_input = window.reshape(1, 60, 1)

    # prediksi hari ke depan (misal 30 hari)
    future_preds = []
    for i in range(30):
        pred = model.predict(X_input)
        future_preds.append(pred[0,0])

        # update window
        X_input = np.append(X_input[:,1:,:], [[pred]], axis=1)

    # inverse transform
    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1,1))

    # buat dataframe tanggal
    last_date = pd.to_datetime(df["Date"].iloc[-1])
    future_dates = [last_date + timedelta(days=i+1) for i in range(30)]

    result = pd.DataFrame({
        "date": future_dates,
        "predicted_close": future_preds.flatten()
    })

    st.subheader("📅 Hasil Prediksi 30 Hari ke Depan")
    st.dataframe(result)

    st.line_chart(result.set_index("date"))
