import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os

st.title("📈 Prediksi Harga Saham TLKM Menggunakan LSTM")
st.write("Aplikasi ini memprediksi harga **Close** saham TLKM dan forecasting 30 hari ke depan.")

# Load dataset
file_path = "daily/TLKM.csv"
df = pd.read_csv(file_path)
st.subheader("📌 Data Awal")
st.dataframe(df.head())

# Preprocessing
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(by='timestamp')
df.reset_index(drop=True, inplace=True)

df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

numeric_cols = ['open', 'high', 'low', 'close', 'volume']
df[numeric_cols] = df[numeric_cols].astype(float)

data = df[['timestamp', 'close']].copy()
data.set_index('timestamp', inplace=True)

train_size = int(len(data) * 0.8)
train = data.iloc[:train_size]
test = data.iloc[train_size:]

st.write(f"🔹 Train: {train.shape} | 🔹 Test: {test.shape}")

# Scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

def create_sequences(dataset, window=60):
    X, y = [], []
    for i in range(window, len(dataset)):
        X.append(dataset[i-window:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

X_all, y_all = create_sequences(scaled_data)
X_train = X_all[:train_size]
y_train = y_all[:train_size]
X_test = X_all[train_size:]
y_test = y_all[train_size:]

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

st.subheader("⚙️ Training Model LSTM")
train_button = st.button("Mulai Training Model")

if train_button:
    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1],1)),
        Dropout(0.2),
        LSTM(50),
        Dense(1)
    ])

    lstm_model.compile(loss='mse', optimizer='adam')
    history = lstm_model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    lstm_pred_scaled = lstm_model.predict(X_test)
    lstm_pred = scaler.inverse_transform(lstm_pred_scaled)
    actual = scaler.inverse_transform(y_test.reshape(-1,1))

    st.success("🚀 Training selesai!")

    # Simpan model dan scaler
    lstm_model.save("lstm_tlkm.h5")
    joblib.dump(scaler, "scaler.pkl")
    st.info("Model & scaler telah disimpan.")

# Prediction Future Days
if os.path.exists("lstm_tlkm.h5"):
    st.subheader("🔮 Prediksi 30 Hari Kedepan (Forecast)")

    # Muat model kalau sudah ada
    from tensorflow.keras.models import load_model
    lstm_model = load_model("lstm_tlkm.h5")
    scaler = joblib.load("scaler.pkl")

    future_days = 30
    last_sequence = scaled_data[-60:]
    current_seq = last_sequence.reshape(1, 60, 1)
    future_predictions = []

    for _ in range(future_days):
        future_pred_scaled = lstm_model.predict(current_seq)[0][0]
        future_predictions.append(float(future_pred_scaled))

        # Update sequence (fix dimensi)
        new_val = np.array(future_pred_scaled).reshape(1, 1, 1)
        current_seq = np.append(current_seq[:, 1:, :], new_val, axis=1)

    future_predictions_actual = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1,1)
    )

    last_date = df['timestamp'].iloc[-1]
    future_dates = pd.date_range(start=last_date, periods=future_days+1, freq='D')[1:]

    future_pred_df = pd.DataFrame({
        "date": future_dates,
        "predicted_close": future_predictions_actual.flatten()
    })

    st.write("### 📄 Tabel Prediksi")
    st.dataframe(future_pred_df)

    st.write("### 📉 Grafik Prediksi")
    plt.figure(figsize=(13,5))
    plt.plot(future_dates, future_predictions_actual, label="LSTM Future Forecast")
    plt.title("Prediksi Harga Penutupan TLKM 30 Hari Kedepan")
    plt.xlabel("Tanggal")
    plt.ylabel("Harga (Rp)")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
