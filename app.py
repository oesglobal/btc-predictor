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
st.caption("Powered by **OESLink** using CoinGecko API")

# ---------------------- Load Model & Scaler ----------------------
model_file = "btc_model.h5"
scaler_file = "btc_scaler.pkl"

if not os.path.exists(model_file) or not os.path.exists(scaler_file):
    st.error("❌ Model or scaler file not found.")
    st.stop()

model = load_model(model_file)
with open(scaler_file, "rb") as f:
    scaler = pickle.load(f)

# ---------------------- Fetch Live Data from CoinGecko ----------------------
@st.cache_data(ttl=60)
@st.cache_data(ttl=60)
def get_coingecko_data(days=1, interval="minutely"):
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": interval
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            st.error(f"❌ CoinGecko API error {response.status_code}")
            st.text(f"Raw response: {response.text}")
            return pd.DataFrame()
        
        data = response.json()
        prices = data.get("prices", [])
        if not prices or not isinstance(prices, list):
            st.warning("⚠️ CoinGecko API returned invalid format.")
            return pd.DataFrame()
        
        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["Date"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["Close"] = df["price"]
        df["Open"] = df["High"] = df["Low"] = df["Close"]
        df["Volume"] = 0.0
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        return df

    except Exception as e:
        st.error(f"🚨 Exception: {e}")
        return pd.DataFrame()

    data = r.json()
    prices = data.get("prices", [])
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["Date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["Close"] = df["price"]
    df["Open"] = df["High"] = df["Low"] = df["Close"]  # Approximation
    df["Volume"] = 0.0  # Not provided by CoinGecko free endpoint
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    return df

df = get_coingecko_data(days=1, interval="minutely")

if df.empty or "Close" not in df.columns:
    st.error("⚠️ CoinGecko API returned no data or missing 'Close' column.")
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

st.subheader("📊 Candlestick Chart (approximate)")
fig = go.Figure(data=[go.Candlestick(
    x=df['Date'],
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    increasing_line_color='green',
    decreasing_line_color='red'
)])
fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Price (USD)",
    height=500,
    xaxis_rangeslider_visible=False,
    template="plotly_dark"
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------- CSV Download ----------------------
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download CSV", csv, "btc_price_data.csv", "text/csv")

# ---------------------- Footer ----------------------
st.markdown("---")
st.markdown("✅ Using [CoinGecko](https://www.coingecko.com/) for public data access.\nNext step: Deploy to `predict.oeslink.one`")
