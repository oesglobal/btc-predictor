import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import requests
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Set page config
st.set_page_config(page_title="BTC Price & Prediction", layout="wide", initial_sidebar_state="collapsed")

# Load LSTM model and scaler
model = load_model('btc_model.h5')
scaler = joblib.load('btc_scaler.pkl')

# Fetch historical data from CoinGecko
@st.cache_data(ttl=300)
def get_btc_data():
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'
    params = {'vs_currency': 'usd', 'days': '2', 'interval': 'minute'}
    res = requests.get(url, params=params)
    data = res.json()
    df = pd.DataFrame(data['prices'], columns=['time', 'price'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    return df

# Predict future price using LSTM
def predict_price(prices):
    sequence_length = 60
    data = scaler.transform(prices[-sequence_length:].reshape(-1, 1))
    X = np.reshape(data, (1, sequence_length, 1))
    predicted = model.predict(X)
    return scaler.inverse_transform(predicted)[0][0]

# Buy/Sell logic
def get_signal(current, predicted):
    if predicted > current * 1.001:
        return "BUY"
    elif predicted < current * 0.999:
        return "SELL"
    else:
        return "HOLD"

# Main
st.markdown("<h1 style='text-align: center; color: #99f9f9;'>📈 Bitcoin (BTC) Live Price & Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #99f9f9;'>Powered by OESLink using CoinGecko API</p>", unsafe_allow_html=True)

df = get_btc_data()
current_price = df['price'].iloc[-1]
predicted_price = predict_price(df['price'].values)

signal = get_signal(current_price, predicted_price)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("💰 Current BTC Price", f"${current_price:,.2f}")
with col2:
    st.metric("📊 Prediction → Signal", signal)
with col3:
    st.metric("🔮 Predicted Price", f"${predicted_price:,.2f}")

# Candlestick chart
fig = go.Figure()

resampled = df.set_index("time").resample('1min').ohlc()['price'].dropna()
fig.add_trace(go.Candlestick(
    x=resampled.index,
    open=resampled['open'],
    high=resampled['high'],
    low=resampled['low'],
    close=resampled['close'],
    increasing_line_color='lime',
    decreasing_line_color='red',
    name='BTC'
))

fig.update_layout(
    title='BTC Live Candlestick Chart (1m)',
    xaxis=dict(
        title=dict(text="Time", font=dict(color="#99f9f9", size=16))
    ),
    yaxis=dict(
        title=dict(text="Price (USD)", font=dict(color="#99f9f9", size=16))
    ),
    plot_bgcolor="#000000",
    paper_bgcolor="#000000",
    font=dict(color="#99f9f9"),
    height=600
)

st.plotly_chart(fig, use_container_width=True)
