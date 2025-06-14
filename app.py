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
st.caption("Powered by **OESLink** using CoinGecko API – live updates every 15 seconds")

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
@st.cache_data(ttl=15)  # Refresh every 15 seconds
@st.cache_data(ttl=15)
def get_btc_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": "1", "interval": "minutely"}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        prices = data.get("prices", [])
        if not prices:
            return pd.DataFrame()

        df = pd.DataFrame(prices, columns=["time", "price"])
        df["Date"] = pd.to_datetime(df["time"], unit="ms")
        df.set_index("Date", inplace=True)

        # ⏳ Resample to 5-minute OHLC
        ohlc = df["price"].resample("5T").ohlc().dropna()
        if ohlc.empty or "close" not in ohlc.columns:
            return pd.DataFrame()

        ohlc["Volume"] = 0.0
        ohlc.reset_index(inplace=True)
        ohlc.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close"
        }, inplace=True)

        return ohlc[["Date", "Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        st.error(f"🚨 Failed to fetch BTC data: {e}")
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

# Price Line Chart
st.subheader("📈 BTC Price Trend (Live)")
line_fig = go.Figure()
line_fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["Close"],
    mode='lines+markers',
    line=dict(color='orange', width=3),
    marker=dict(size=3),
    name='Close Price'
))
line_fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Price (USD)",
    height=400,
    template="plotly_dark",
    margin=dict(l=10, r=10, t=20, b=20),
    showlegend=False
)
st.plotly_chart(line_fig, use_container_width=True)

# Candlestick Chart
btc_df = get_btc_data()

if btc_df.empty:
    st.warning("⚠️ Live BTC data is unavailable or missing.")
else:
    st.subheader("📊 Candlestick Chart (5-min candles – more visible)")
    fig = go.Figure(data=[go.Candlestick(
        x=btc_df["Date"],
        open=btc_df["Open"],
        high=btc_df["High"],
        low=btc_df["Low"],
        close=btc_df["Close"],
        increasing_line_color='lime',
        decreasing_line_color='red',
        increasing_line_width=5,
        decreasing_line_width=5,
        whiskerwidth=0.7
    )])
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        height=550,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=20, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

# CSV Download
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download BTC CSV", csv, "btc_price_data.csv", "text/csv")

# Footer
st.markdown("---")
st.markdown("✅ Data from [CoinGecko](https://www.coingecko.com/) – auto-updates every **15 seconds**.\n🔁 To see live price movement, leave this page open.\nDeploy to: **`https://predict.oeslink.one`**")
