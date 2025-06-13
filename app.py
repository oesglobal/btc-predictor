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

# ----- Load LSTM Model and Scaler -----
@st.cache_resource(show_spinner=False)
def load_lstm_model_and_scaler():
    model = load_model("btc_model.h5")
    with open("btc_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_lstm_model_and_scaler()

# ----- Fetch BTC price data from CoinGecko -----
def fetch_btc_price():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    try:
        response = requests.get(url)
        price = response.json()["bitcoin"]["usd"]
        return price
    except:
        return None

# ----- Fetch historical OHLC data from CoinGecko (1m interval last 100 points) -----
@st.cache_data(ttl=60)
def fetch_ohlc_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc?vs_currency=usd&days=1"
    try:
        response = requests.get(url)
        data = response.json()
        # data = [ [timestamp, open, high, low, close], ...]
        df = pd.DataFrame(data, columns=["Timestamp", "Open", "High", "Low", "Close"])
        df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
        return df
    except:
        return pd.DataFrame()

# ----- Prepare data for prediction -----
def prepare_data_for_prediction(df):
    closes = df["Close"].values.reshape(-1,1)
    scaled = scaler.transform(closes)
    sequence = scaled[-60:]  # last 60 points
    X_test = np.array([sequence])
    return X_test

# ----- Predict next price -----
def predict_next_price(X):
    pred_scaled = model.predict(X)
    pred_price = scaler.inverse_transform(pred_scaled)
    return pred_price[0][0]

# ----- Main app UI -----
st.title("📈 Bitcoin (BTC) Live Price & Prediction")
st.markdown("Powered by **OESLink** using CoinGecko API")

current_price = fetch_btc_price()

if current_price:
    st.markdown(f"### 💰 Current BTC Price\n\n**{current_price:,.2f} USD**")
else:
    st.markdown("⚠️ Failed to fetch current BTC price.")

# Fetch OHLC data and display charts
df_ohlc = fetch_ohlc_data()

if not df_ohlc.empty:
    X_test = prepare_data_for_prediction(df_ohlc)
    predicted_price = predict_next_price(X_test)
    
    # Simple buy/sell logic: if predicted price > current price -> BUY, else SELL
    signal = "BUY" if predicted_price > current_price else "SELL"
    st.markdown(f"### 📊 Prediction → **{signal} Signal**")
    st.markdown(f"### Predicted Price: **{predicted_price:,.2f} USD**")

    # Price Line Chart
    st.subheader("📈 Price Line Chart (Auto-updated)")
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=df_ohlc["Date"],
        y=df_ohlc["Close"],
        mode='lines',
        name='Close Price',
        line=dict(color='#00ffff', width=2)
    ))
    fig_line.update_layout(
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font_color='#99f9f9',
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(title="Time", titlefont=dict(color='#99f9f9'), tickfont=dict(color='#99f9f9')),
        yaxis=dict(title="Price (USD)", titlefont=dict(color='#99f9f9'), tickfont=dict(color='#99f9f9')),
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # Candlestick Chart
    st.subheader("📊 Live Candlestick Chart (1m Interval)")
    fig_candle = go.Figure(data=[go.Candlestick(
        x=df_ohlc['Date'],
        open=df_ohlc['Open'],
        high=df_ohlc['High'],
        low=df_ohlc['Low'],
        close=df_ohlc['Close'],
        increasing_line_color='#00ff00',
        decreasing_line_color='#ff5555',
        line_width=2
    )])
    fig_candle.update_layout(
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font_color='#99f9f9',
        height=600,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(title='Time', titlefont=dict(color='#99f9f9'), tickfont=dict(color='#99f9f9'), rangeslider=dict(visible=False)),
        yaxis=dict(title='Price (USD)', titlefont=dict(color='#99f9f9'), tickfont=dict(color='#99f9f9')),
    )
    st.plotly_chart(fig_candle, use_container_width=True)

else:
    st.warning("Failed to load OHLC data for charts.")

st.markdown("---")
st.markdown("✅ Using CoinGecko for public data access. Next step: Deploy to predict.oeslink.one")
