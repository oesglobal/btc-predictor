import streamlit as st
import pandas as pd
import requests
import time
import plotly.graph_objs as go
from datetime import datetime

# === App Configuration ===
st.set_page_config(page_title="BTC Live Price & Prediction", layout="wide")
st.title("📈 Bitcoin (BTC) Live Price & Prediction")
st.caption("Powered by OESLink using CoinGecko API – live updates every 15 seconds")

# === Function to Fetch Live BTC/USDT Price Data ===
@st.cache_data(ttl=15)
def fetch_btc_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": "1", "interval": "minute"}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        prices = response.json().get("prices", [])
        df = pd.DataFrame(prices, columns=["Timestamp", "Price"])
        df["DateTime"] = pd.to_datetime(df["Timestamp"], unit="ms")
        df.set_index("DateTime", inplace=True)
        df["Open"] = df["Price"].shift(1)
        df["Close"] = df["Price"]
        df["High"] = df[["Open", "Close"]].max(axis=1)
        df["Low"] = df[["Open", "Close"]].min(axis=1)
        return df[["Open", "High", "Low", "Close"]].dropna()
    except Exception as e:
        st.error(f"⚠️ Failed to fetch BTC price data: {e}")
        return pd.DataFrame()

# === Load Data ===
df = fetch_btc_data()
if df.empty or "Close" not in df.columns:
    st.warning("⚠️ CoinGecko API returned no data or missing 'Close' column.")
else:
    latest_price = df["Close"].iloc[-1]
    st.metric(label="Current BTC/USDT Price", value=f"${latest_price:,.2f}")

    # === Plot Candlestick Chart ===
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        increasing_line_color='green',
        decreasing_line_color='red',
        line=dict(width=2)
    )])
    fig.update_layout(
        title="Live BTC/USDT Candlestick Chart (1-min interval)",
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # === CSV Export Option ===
    csv = df.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download BTC Data as CSV", csv, "btc_data.csv", "text/csv")
