import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import os
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from datetime import datetime
from binance.client import Client

# ---------------------- Page Config ----------------------
st.set_page_config(page_title="Bitcoin Predictor", layout="wide")
st.title("\U0001F4C8 Bitcoin (BTC) Live Price & Prediction")
st.caption("Powered by **OESLink**")

# ---------------------- Binance API Setup ----------------------
API_KEY = st.secrets["BINANCE_API_KEY"] if "BINANCE_API_KEY" in st.secrets else ""
API_SECRET = st.secrets["BINANCE_API_SECRET"] if "BINANCE_API_SECRET" in st.secrets else ""

binance_client = None
if API_KEY and API_SECRET:
    import asyncio
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    binance_client = Client(API_KEY, API_SECRET)

# ---------------------- Load Model & Scaler ----------------------
model_file = "btc_model.h5"
scaler_file = "btc_scaler.pkl"

if not os.path.exists(model_file) or not os.path.exists(scaler_file):
    st.error("Model or scaler file not found.")
    st.stop()

model = load_model(model_file)model = load_model(model_file)
model.compile(optimizer='adam', loss='mse') 
with open(scaler_file, "rb") as f:
    scaler = pickle.load(f)

# ---------------------- Fetch Live Data ----------------------
def get_binance_data(symbol="BTCUSDT", interval="1m", limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()
    if not data or not isinstance(data, list):
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=[
        "Open time", "Open", "High", "Low", "Close", "Volume",
        "Close time", "Quote asset volume", "Number of trades",
        "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
    ])
    df = df[["Open time", "Open", "High", "Low", "Close", "Volume"]]
    df["Open time"] = pd.to_datetime(df["Open time"], unit='ms')
    df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df.rename(columns={"Open time": "Date"}, inplace=True)
    return df

df = get_binance_data("BTCUSDT")

if df.empty or "Close" not in df.columns:
    st.warning("\u26A0\uFE0F Binance API returned no data or missing 'Close' column.")
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
    st.metric("\U0001FA99 Current BTC/USDT Price", f"{current_price:,.2f} USDT")

with col2:
    st.metric(f"\U0001F4CA Prediction → {signal} Signal", f"{predicted_price:,.2f} USDT")

# ---------------------- Line Chart ----------------------
st.line_chart(df.set_index("Date")["Close"], use_container_width=True)

# ---------------------- Candlestick Chart ----------------------
st.subheader("📊 Live Candlestick Chart (1m interval)")
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
    yaxis_title="Price (USDT)",
    height=500,
    xaxis_rangeslider_visible=False,
    template="plotly_dark"
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------- CSV Export ----------------------
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("\U0001F4C5 Download Price Data", csv, "btc_price_data.csv", "text/csv")

# ---------------------- Auto-Trading (Optional) ----------------------
if binance_client:
    st.subheader("\U0001F916 Auto-Trading Bot (BTC Only)")
    trade_amount = st.number_input("Amount to trade (USDT)", min_value=10.0, step=10.0)
    if st.button("Place Order Automatically"):
        try:
            order = binance_client.order_market_buy(
                symbol="BTCUSDT",
                quoteOrderQty=str(trade_amount)
            ) if signal == "BUY" else binance_client.order_market_sell(
                symbol="BTCUSDT",
                quoteOrderQty=str(trade_amount)
            )
            st.success(f"\u2705 {signal} order placed! Order ID: {order['orderId']}")
        except Exception as e:
            st.error(f"\u274C Error placing order: {e}")
else:
    st.info("\U0001F512 Add your Binance API keys to `.streamlit/secrets.toml` to enable trading")

# ---------------------- Footer ----------------------
st.markdown("---")
st.markdown("Next: Add sentiment analysis \U0001F50D + deploy to `predict.oeslink.one` \U0001F310")
