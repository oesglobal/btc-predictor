import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import os
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from datetime import datetime

# Config
st.set_page_config(page_title="Bitcoin Predictor", layout="wide")
st.title("📈 Bitcoin (BTC) Live Price & Prediction")
st.caption("Powered by **OESLink** using CoinGecko API")

# Load model & scaler
model_file = "btc_model.h5"
scaler_file = "btc_scaler.pkl"

if not os.path.exists(model_file) or not os.path.exists(scaler_file):
    st.error("❌ Model or scaler file not found.")
    st.stop()

model = load_model(model_file)
with open(scaler_file, "rb") as f:
    scaler = pickle.load(f)

# Fetch BTC data from CoinGecko
@st.cache_data(ttl=60)
def get_btc_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": "1"}
    try:
        response = requests.get(url, params=params)
        data = response.json()
        prices = data.get("prices", [])
        if not prices:
            return pd.DataFrame()

        df = pd.DataFrame(prices, columns=["time", "price"])
        df["Date"] = pd.to_datetime(df["time"], unit="ms")
        df["Close"] = df["price"]
        df["Open"] = df["High"] = df["Low"] = df["Close"]
        df["Volume"] = 0.0
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        return df

    except Exception as e:
        st.error(f"🚨 Failed to fetch data from CoinGecko: {e}")
        return pd.DataFrame()

# Load data
df = get_btc_data()
if df.empty or "Close" not in df.columns:
    st.error("⚠️ CoinGecko API returned no data or missing 'Close' column.")
    st.stop()

# Predict next price
def predict_next(df, model, scaler, sequence_length=60):
    df_scaled = scaler.transform(df[["Close"]])
    last_sequence = df_scaled[-sequence_length:]
    X_input = np.reshape(last_sequence, (1, sequence_length, 1))
    prediction = model.predict(X_input)
    return scaler.inverse_transform(prediction)[0][0]

current_price = df["Close"].iloc[-1]
predicted_price = predict_next(df, model, scaler)
signal = "BUY" if predicted_price > current_price else "SELL"

# Display metrics
col1, col2 = st.columns(2)
col1.metric("💰 Current BTC Price", f"{current_price:,.2f} USD")
col2.metric(f"📊 Prediction → {signal} Signal", f"{predicted_price:,.2f} USD")

# Line chart
st.subheader("📈 Price Line Chart")
st.line_chart(df.set_index("Date")["Close"])

# Candlestick chart
st.subheader("📊 Candlestick Chart (approximate)")
fig = go.Figure(data=[go.Candlestick(
    x=df["Date"],
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    increasing_line_color='green',
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

# CSV download
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download CSV", csv, "btc_price_data.csv", "text/csv")

# Footer
st.markdown("---")
st.markdown("✅ Using [CoinGecko](https://www.coingecko.com/) for public data access.\nNext step: Deploy to `predict.oeslink.one`")
