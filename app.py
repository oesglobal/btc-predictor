import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import requests
import time
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="BTC Predictor", layout="wide")

# Load model and scaler
@st.cache_resource(show_spinner=False)
def load_lstm_model_and_scaler():
    model = load_model("btc_model.h5")
    with open("btc_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_lstm_model_and_scaler()

# Fetch BTC price from CoinGecko
def fetch_btc_price():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    try:
        response = requests.get(url)
        price = response.json()["bitcoin"]["usd"]
        return price
    except:
        return None

# Fetch OHLC data from CoinGecko
@st.cache_data(ttl=60)
def fetch_ohlc_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc?vs_currency=usd&days=1"
    try:
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data, columns=["Timestamp", "Open", "High", "Low", "Close"])
        df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
        return df
    except:
        return pd.DataFrame()

# Prepare input sequence
def prepare_data_for_prediction(df):
    closes = df["Close"].values.reshape(-1,1)
    scaled = scaler.transform(closes)
    sequence = scaled[-60:]
    X_test = np.array([sequence])
    return X_test

# Predict next price
def predict_next_price(X):
    pred_scaled = model.predict(X)
    pred_price = scaler.inverse_transform(pred_scaled)
    return pred_price[0][0]

# App layout
st.title("📈 Bitcoin (BTC) Live Price & Prediction")
st.markdown("Powered by **OESLink** using CoinGecko API")

current_price = fetch_btc_price()

if current_price:
    st.markdown(f"### 💰 Current BTC Price\n\n**{current_price:,.2f} USD**")
else:
    st.warning("⚠️ Failed to fetch current BTC price.")

df_ohlc = fetch_ohlc_data()

if not df_ohlc.empty:
    X_test = prepare_data_for_prediction(df_ohlc)
    predicted_price = predict_next_price(X_test)

    signal = "BUY" if predicted_price > current_price else "SELL"
    st.markdown(f"### 📊 Prediction → **{signal} Signal**")
    st.markdown(f"### Predicted Price: **{predicted_price:,.2f} USD**")

    # Line Chart
    st.subheader("📈 Price Line Chart")
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=df_ohlc["Date"],
        y=df_ohlc["Close"],
        mode='lines',
        name='Close Price',
        line=dict(color='cyan', width=2)
    ))
    st.plotly_chart(fig_line, use_container_width=True)

    # Candlestick Chart
    st.subheader("📊 Live Candlestick Chart (1m Interval)")
    fig_candle = go.Figure(data=[go.Candlestick(
        x=df_ohlc['Date'],
        open=df_ohlc['Open'],
        high=df_ohlc['High'],
        low=df_ohlc['Low'],
        close=df_ohlc['Close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])
    fig_candle.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_candle, use_container_width=True)

else:
    st.warning("⚠️ Unable to load OHLC data.")

st.markdown("---")
st.markdown("✅ Using CoinGecko for public data access. Next step: Deploy to **predict.oeslink.one**")
