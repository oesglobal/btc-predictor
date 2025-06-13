
import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import os
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from datetime import datetime

# ---------------------- Config ----------------------
st.set_page_config(page_title="Bitcoin Predictor", layout="wide")
st.title("📈 Bitcoin (BTC) Live Price & Prediction")
st.caption("Powered by **OESLink** using Binance API")

# ---------------------- Load Model & Scaler ----------------------
model_file = "btc_model.h5"
scaler_file = "btc_scaler.pkl"

if not os.path.exists(model_file) or not os.path.exists(scaler_file):
    st.error("❌ Model or scaler file not found.")
    st.stop()

model = load_model(model_file)
with open(scaler_file, "rb") as f:
    scaler = pickle.load(f)

# ---------------------- Fetch Live Data from Binance ----------------------
@st.cache_data(ttl=15)
@st.cache_data(ttl=15)
def get_binance_data():
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "15s",
        "limit": 100
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if not isinstance(data, list) or len(data) == 0:
            st.warning("⚠️ Binance API returned no data.")
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=[
            "Open Time", "Open", "High", "Low", "Close", "Volume",
            "Close Time", "Quote Asset Volume", "Number of Trades",
            "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
        ])
        df["Date"] = pd.to_datetime(df["Open Time"], unit="ms")
        df["Open"] = df["Open"].astype(float)
        df["High"] = df["High"].astype(float)
        df["Low"] = df["Low"].astype(float)
        df["Close"] = df["Close"].astype(float)
        df["Volume"] = df["Volume"].astype(float)
        return df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    except Exception as e:
        st.error(f"🚨 Exception fetching from Binance: {e}")
        return pd.DataFrame()


# ---------------------- Load Data ----------------------
df = get_binance_data()

if df.empty or "Close" not in df.columns:
    st.error("⚠️ Binance API returned no data or missing 'Close' column.")
    st.stop()

# ---------------------- Predict Next Price ----------------------
def predict_next(df, model, scaler, sequence_length=60):
    df_scaled = scaler.transform(df[["Close"]])
    last_sequence = df_scaled[-sequence_length:]
    X_input = np.reshape(last_sequence, (1, sequence_length, 1))
    prediction = model.predict(X_input)
    predicted_price = scaler.inverse_transform(prediction)[0][0]
    return predicted_price

# ---------------------- Signal Logic ----------------------
current_price = df["Close"].iloc[-1]
predicted_price = predict_next(df, model, scaler)
signal = "BUY" if predicted_price > current_price else "SELL"

# ---------------------- Display ----------------------
col1, col2 = st.columns(2)
with col1:
    st.metric("💰 Current BTC Price", f"{current_price:,.2f} USD")
with col2:
    st.metric(f"📊 Prediction → {signal} Signal", f"{predicted_price:,.2f} USD")

# ---------------------- Charts ----------------------
st.subheader("📈 Price Line Chart")
st.line_chart(df.set_index("Date")["Close"], use_container_width=True)

st.subheader("📊 Live Candlestick Chart (15s auto-update)")
fig = go.Figure(data=[go.Candlestick(
    x=df['Date'],
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    increasing_line_color='lime',
    decreasing_line_color='red'
)])
fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Price (USD)",
    height=550,
    xaxis_rangeslider_visible=False,
    template="plotly_dark"
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------- CSV Download ----------------------
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download CSV", csv, "btc_price_data.csv", "text/csv")

# ---------------------- Footer ----------------------
st.markdown("---")
st.markdown("✅ Using [Binance API](https://www.binance.com/en) for 15-second live data updates.\nNext step: Deploy to `predict.oeslink.one`")
