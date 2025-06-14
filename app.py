import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import os
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

# ---------------------- Config ----------------------
st.set_page_config(page_title="Bitcoin Predictor", layout="wide")
st.title("📈 Bitcoin (BTC) Live Price & Prediction")
st.caption("Powered by **OESLink** using CoinMarketCap API")

# ---------------------- Load Model & Scaler ----------------------
model_file = "btc_model.h5"
scaler_file = "btc_scaler.pkl"

if not os.path.exists(model_file) or not os.path.exists(scaler_file):
    st.error("❌ Model or scaler file not found.")
    st.stop()

model = load_model(model_file)
with open(scaler_file, "rb") as f:
    scaler = pickle.load(f)

# ---------------------- API Key & Headers ----------------------
CMC_API_KEY = "048d9f91-1d4a-49aa-89a2-f2fb86af26b1"  # <<-- REPLACE THIS!
headers = {
    "Accepts": "application/json",
    "X-CMC_PRO_API_KEY": CMC_API_KEY,
}

# ---------------------- Fetch BTC Historical Data ----------------------
@st.cache_data(ttl=60)
def get_btc_data():
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"
    params = {
        "symbol": "BTC",
        "convert": "USD",
        "time_period": "hourly",
        "interval": "5m",
        "count": 24,
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        quotes = data["data"]["quotes"]

        df = pd.DataFrame([{
            "Date": pd.to_datetime(q["timestamp"]),
            "Open": q["quote"]["USD"]["open"],
            "High": q["quote"]["USD"]["high"],
            "Low": q["quote"]["USD"]["low"],
            "Close": q["quote"]["USD"]["close"],
            "Volume": q["quote"]["USD"]["volume"]
        } for q in quotes])

        return df
    except Exception as e:
        st.error(f"🚨 Failed to fetch BTC data from CoinMarketCap: {e}")
        return pd.DataFrame()

# ---------------------- Load & Filter Data ----------------------
df = get_btc_data()

if df.empty or "Close" not in df.columns:
    st.error("⚠️ No data returned or missing 'Close' column.")
    st.stop()

# Filter for last 2 hours
df = df[df["Date"] > (datetime.utcnow() - timedelta(hours=2))]

# ---------------------- Predict Next Price ----------------------
def predict_next(df, model, scaler, sequence_length=60):
    close_scaled = scaler.transform(df[["Close"]])
    last_sequence = close_scaled[-sequence_length:]
    X_input = np.reshape(last_sequence, (1, sequence_length, 1))
    prediction = model.predict(X_input)
    return scaler.inverse_transform(prediction)[0][0]

current_price = df["Close"].iloc[-1]
predicted_price = predict_next(df, model, scaler)
signal = "BUY" if predicted_price > current_price else "SELL"

# ---------------------- Display Metrics ----------------------
col1, col2 = st.columns(2)
col1.metric("💰 Current BTC Price", f"{current_price:,.2f} USD")
col2.metric(f"📊 Prediction → {signal} Signal", f"{predicted_price:,.2f} USD")

# ---------------------- Line Chart ----------------------
st.subheader("📈 Price Line Chart (last 2 hours)")
st.line_chart(df.set_index("Date")["Close"])

# ---------------------- Candlestick Chart ----------------------
st.subheader("📊 Candlestick Chart (zoomed view)")
fig = go.Figure(data=[go.Candlestick(
    x=df["Date"],
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    increasing_line_color='limegreen',
    decreasing_line_color='orangered',
    increasing_fillcolor='limegreen',
    decreasing_fillcolor='orangered'
)])
fig.update_layout(
    xaxis_title="Time (UTC)",
    yaxis_title="BTC Price (USD)",
    xaxis_rangeslider_visible=False,
    height=600,
    template="plotly_dark"
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------- CSV Export ----------------------
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download CSV", csv, "btc_price_data.csv", "text/csv")

# ---------------------- Footer ----------------------
st.markdown("---")
st.markdown("✅ Using [CoinMarketCap](https://coinmarketcap.com/) for live BTC data.\nDeployed by **OESLink** • All rights reserved.")
