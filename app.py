import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objs as go
from keras.models import load_model
import pickle
from datetime import datetime
import time

# Load model and scaler
model = load_model('btc_model.h5')
with open('btc_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="BTC Predictor", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; color: #99f9f9;'>📈 Bitcoin (BTC) Live Price & Prediction</h1>
    <h4 style='text-align: center; color: #99f9f9;'>Powered by OESLink using CoinGecko API</h4>
    """,
    unsafe_allow_html=True
)

@st.cache_data(ttl=60)
@st.cache_data(ttl=60)
def get_btc_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": "1", "interval": "minutely"}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "prices" not in data:
            st.error("⚠️ CoinGecko API response missing 'prices'. Try again later.")
            return None

        df = pd.DataFrame(data["prices"], columns=["time", "price"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        return df

    except requests.exceptions.RequestException as e:
        st.error(f"🚨 Failed to fetch data from CoinGecko: {e}")
        return None


    last_sequence = scaled_data[-sequence_length:]
    X_test = np.array([last_sequence])
    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)
    return float(df['price'].iloc[-1]), float(predicted_price[0][0])

df = get_btc_data()

if df is not None and len(df) >= 60:
    current_price, predicted_price = predict_price(df)

    signal = "BUY" if predicted_price > current_price else "SELL"
    signal_color = "#00FF00" if signal == "BUY" else "#FF3333"

    st.markdown(f"""
    <div style='color: #99f9f9; font-size: 20px;'>
    💰 <b>Current BTC Price</b><br>
    <span style='font-size: 30px;'>{current_price:,.2f} USD</span><br><br>
    📊 <b>Prediction →</b> <span style='color: {signal_color}; font-weight: bold;'>{signal} Signal</span><br>
    Predicted Price: {predicted_price:,.2f} USD
    </div><br>
    """, unsafe_allow_html=True)

    # Price Line Chart
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df["time"], y=df["price"], mode='lines', name='Price'))
    fig1.update_layout(
        title='📈 Price Line Chart',
        xaxis_title='Time',
        yaxis_title='USD',
        template='plotly_dark',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='#99f9f9')
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Live Candlestick Chart
    df_candle = df.copy()
    df_candle.set_index("time", inplace=True)
    df_candle["open"] = df_candle["price"].shift(1)
    df_candle["high"] = df_candle["price"].rolling(window=3).max()
    df_candle["low"] = df_candle["price"].rolling(window=3).min()
    df_candle["close"] = df_candle["price"]

    fig2 = go.Figure(data=[go.Candlestick(
        x=df_candle.index,
        open=df_candle["open"],
        high=df_candle["high"],
        low=df_candle["low"],
        close=df_candle["close"],
        increasing_line_color='lime', decreasing_line_color='red'
    )])

    fig2.update_layout(
        title='📊 Live Candlestick Chart (1m Interval)',
        xaxis_title='Time',
        yaxis_title='USD',
        template='plotly_dark',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='#99f9f9')
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<br><hr><div style='text-align:center; color:#99f9f9;'>✅ Using CoinGecko for public data access. Next step: Deploy to <b>predict.oeslink.one</b></div>", unsafe_allow_html=True)

else:
    st.warning("Not enough data to predict. Please wait for more data or check your internet connection.")
