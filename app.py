import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests

# ---------------------- Config ----------------------
st.set_page_config(page_title="Bitcoin Predictor", layout="wide")
st.title("📈 Bitcoin (BTC) Live Price & Prediction")
st.caption("Powered by **OESLink** using Binance API")

# ---------------------- Fetch Live Data from Binance ----------------------
@st.cache_data(ttl=15)
# Replace this:
# df = get_binance_data()
# With this:
df = get_coingecko_data()
():
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "1m",
        "limit": 200
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # Parse into DataFrame
        df = pd.DataFrame(data, columns=[
            "Open Time", "Open", "High", "Low", "Close", "Volume",
            "Close Time", "Quote Asset Volume", "Number of Trades",
            "Taker Buy Base Vol", "Taker Buy Quote Vol", "Ignore"
        ])
        df["Date"] = pd.to_datetime(df["Open Time"], unit="ms")
        df[["Open","High","Low","Close","Volume"]] = df[["Open","High","Low","Close","Volume"]].astype(float)
        return df[["Date","Open","High","Low","Close","Volume"]]
    except Exception as e:
        st.error(f"🚨 Failed to fetch Binance data: {e}")
        return pd.DataFrame()

# ---------------------- Load Data ----------------------
df = get_binance_data()
if df.empty:
    st.error("⚠️ Binance API returned no data.")
    st.stop()

# ---------------------- Predict Next Price ----------------------
# (Assumes you have btc_model.h5 and btc_scaler.pkl in the same folder)
import pickle
from tensorflow.keras.models import load_model
model = load_model("btc_model.h5")
with open("btc_scaler.pkl","rb") as f:
    scaler = pickle.load(f)

def predict_next(df, sequence_length=60):
    arr = scaler.transform(df[["Close"]])
    seq = arr[-sequence_length:]
    X = seq.reshape((1, sequence_length, 1))
    pred = model.predict(X)
    return float(scaler.inverse_transform(pred)[0][0])

current_price = df["Close"].iloc[-1]
predicted_price = predict_next(df)
signal = "BUY" if predicted_price > current_price else "SELL"

# ---------------------- Display ----------------------
col1, col2 = st.columns(2)
col1.metric("💰 Current BTC Price", f"{current_price:,.2f} USD")
col2.metric("📊 Prediction → Signal", signal, f"{predicted_price:,.2f} USD")

# ---------------------- Charts ----------------------
st.subheader("📈 Price Line Chart")
st.line_chart(df.set_index("Date")["Close"])

st.subheader("📊 Live Candlestick Chart")
fig = go.Figure(data=[go.Candlestick(
    x=df["Date"],
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    increasing_line_color="green",
    decreasing_line_color="red",
    line_width=2
)])
fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Price (USD)",
    height=600,
    xaxis_rangeslider_visible=False,
    template="plotly_dark"
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------- CSV Download ----------------------
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download CSV", csv, "btc_price_data.csv", "text/csv")

# ---------------------- Footer ----------------------
st.markdown("---")
st.markdown("✅ Using [Binance API](https://www.binance.com/) for real-time data (1m candles, refreshed every 15s).")
