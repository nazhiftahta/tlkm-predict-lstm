# 📈 Stock Price Prediction with LSTM & Streamlit

Project ini merupakan aplikasi machine learning untuk memprediksi harga saham 
menggunakan model **Long Short-Term Memory (LSTM)**.  
Aplikasi ini dibuat menggunakan **TensorFlow/Keras**, **Scikit-Learn**, dan 
dideploy dalam bentuk web menggunakan **Streamlit**.

---

## 🚀 Features
- Data preprocessing (MinMaxScaler, sequence generation)
- Training model LSTM untuk prediksi harga saham
- Load model `.h5` dan scaler `.pkl`
- Visualisasi grafik harga aktual vs prediksi
- Input prediksi via web-app Streamlit
- Deployment lokal melalui `streamlit run app.py`

---

## 📂 Project Structure
📁 CapstoneProject/
│── app.py
│── model_lstm.h5
│── scaler.pkl
│── DaftarSaham.csv 
│── CapstoneProject.ipynb
│── daily/
│ └── TLKM.csv
│── hourly/
│ └── TLKM.csv
│── minutes/
│ └── TLKM.csv
│── README.md


---

## 🧠 Model Explanation

Model LSTM digunakan karena mampu menangkap pola sekuensial 
dalam data time series seperti harga saham.  
Beberapa tahapan training:

1. Load dataset
2. Scaling fitur menggunakan `MinMaxScaler`
3. Membentuk data sequence
4. Membuat arsitektur LSTM
5. Training model
6. Menyimpan model & scaler

---

## 🛠 Installation & Setup

### 1. Clone repository

```bash
git clone https://github.com/USERNAME/REPOSITORY-NAME.git
cd REPOSITORY-NAME
