# 8
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional

st.set_page_config(page_title="LSTM Stock Predictor", layout="wide")
st.title("📈Stock Price Prediction Dashboard")

# --- Theme Toggle ---
theme_choice = st.radio("🎨 Select Theme:", ["Dark Mode", "Light Mode"], horizontal=True)
plotly_theme = "plotly_dark" if theme_choice == "Dark Mode" else "plotly_white"

# --- Inject Custom Theme Styles for Better Visibility ---
font_url = "https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap"
st.markdown(f'<link href="{font_url}" rel="stylesheet">', unsafe_allow_html=True)

dark_css = """
<style>
html, body, [class*="css"] {
    background-color: #121212 !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif;
}
.stApp {
    background-color: #121212 !important;
}
h1, h2, h3, h4, h5, h6, label, .stRadio label, .stSelectbox label, .stMetricLabel {
    color: #ffffff !important;
}
[data-testid="stMetric"] {
    background-color: #1e1e1e !important;
    padding: 8px;
    border-radius: 10px;
}
[data-testid="stMetricLabel"] {
    color: #00e0ff !important;
    font-weight: 700;
    font-size: 16px;
    opacity: 1 !important;
}
[data-testid="stMetricValue"] {
    color: #00ffff !important;
    font-weight: bold;
    font-size: 28px;
}
input, .stTextInput > div > div > input, .stDateInput input {
    background-color: #1e1e1e !important;
    color: #ffffff !important;
    border: 1.5px solid #00e0ff !important;
    font-size: 16px !important;
}
.stDateInput label, .stTextInput label {
    color: #00e0ff !important;
    font-weight: 600 !important;
    font-size: 16px !important;
}
.stSelectbox div[data-baseweb="select"] {
    background-color: #1e1e1e !important;
    color: #ffffff !important;
}
.stButton > button {
    background-color: #333 !important;
    color: #ffffff !important;
}
</style>
"""

light_css = """
<style>
html, body, [class*="css"] {
    background-color: #f9f9f9 !important;
    color: #1e1e1e !important;
    font-family: 'Inter', sans-serif;
}
.stApp {
    background-color: #ffffff !important;
}
[data-testid="stMetric"] {
    background-color: #f1f5f9 !important;
    padding: 8px;
    border-radius: 10px;
}
[data-testid="stMetricLabel"] {
    color: #0ea5e9 !important;
    font-weight: 700;
    font-size: 16px;
    opacity: 1 !important;
}
[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-weight: bold;
    font-size: 28px;
}
input, .stTextInput > div > div > input, .stDateInput input {
    background-color: #f1f5f9 !important;
    color: #1e1e1e !important;
    border: 1.5px solid #0ea5e9 !important;
    font-size: 16px !important;
}
.stDateInput label, .stTextInput label {
    color: #0ea5e9 !important;
    font-weight: 600 !important;
    font-size: 16px !important;
}
.stSelectbox div[data-baseweb="select"] {
    background-color: #f1f5f9 !important;
    color: #1e1e1e !important;
}
.stButton > button {
    background-color: #0ea5e9 !important;
    color: #ffffff !important;
}
</style>
"""

st.markdown(dark_css if theme_choice == "Dark Mode" else light_css, unsafe_allow_html=True)

# --- Dropdown Ticker Selection ---
ticker_options = {
    "SBIN.NS": "State Bank of India",
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "ITC.NS": "ITC Limited",
    "WIPRO.NS": "Wipro",
    "AXISBANK.NS": "Axis Bank",
    "KOTAKBANK.NS": "Kotak Mahindra Bank"
}

selected_label = st.selectbox(
    "Select NSE Ticker:",
    options=[f"{v} ({k})" for k, v in ticker_options.items()],
    index=0
)
ticker = selected_label.split("(")[-1].strip(")")

# --- Load Data ---
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start="2015-01-01")
    return data.dropna()

data = load_data(ticker)

# --- Live Info ---
try:
    info = yf.Ticker(ticker).info
    st.subheader("💹 Live Stock Price Overview")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Current Price", f"₹{info.get('currentPrice', 0):.2f}")
    col2.metric("Previous Close", f"₹{info.get('previousClose', 0):.2f}")
    col3.metric("Open", f"₹{info.get('open', 0):.2f}")
    col4.metric("Day High", f"₹{info.get('dayHigh', 0):.2f}")
    col5.metric("Day Low", f"₹{info.get('dayLow', 0):.2f}")
    col6.metric("Symbol", ticker.upper())
except:
    st.warning("⚠️ Unable to fetch live stock info.")

if data.empty or "Close" not in data.columns:
    st.error("❌ No stock data found.")
    st.stop()

# --- Preprocessing ---
scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(data[["Close"]])

seq_len = 60
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_close, seq_len)
X = X.reshape((X.shape[0], X.shape[1], 1))

# --- Train/Test Split ---
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# --- BiLSTM Model ---
model = Sequential([
    Bidirectional(LSTM(60, return_sequences=True), input_shape=(seq_len, 1)),
    Bidirectional(LSTM(60)),
    Dense(1)
])
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

# --- Prediction ---
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(y_test)

# --- Actual vs Predicted Chart ---
st.subheader("📉 Actual vs Predicted Closing Prices")
dates = data.index[-len(predictions):]
fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x=dates, y=actual.flatten(), name="Actual", line=dict(color="lime")))
fig_pred.add_trace(go.Scatter(x=dates, y=predictions.flatten(), name="Predicted", line=dict(color="orange", dash="dot")))
fig_pred.update_layout(template=plotly_theme, xaxis_title="Date", yaxis_title="Price (₹)")
st.plotly_chart(fig_pred, use_container_width=True)

# --- Forecast 7 Days ---
st.subheader("🔮 Forecast: Next 7 Days")
last_seq = scaled_close[-seq_len:]
future_scaled = []
for _ in range(7):
    pred = model.predict(last_seq.reshape(1, seq_len, 1))[0][0]
    future_scaled.append(pred)
    last_seq = np.append(last_seq[1:], [[pred]], axis=0)

future_prices = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()
future_dates = [data.index[-1] + timedelta(days=i + 1) for i in range(7)]
future_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": future_prices})

fig_future = go.Figure()
fig_future.add_trace(go.Scatter(
    x=future_df["Date"],
    y=future_df["Predicted_Close"],
    mode="lines+markers",
    line=dict(color="cyan"),
    name="Predicted Close"
))
fig_future.update_layout(template=plotly_theme, xaxis_title="Date", yaxis_title="Predicted Price")
st.plotly_chart(fig_future, use_container_width=True)
st.dataframe(future_df.style.format({"Predicted_Close": "₹{:.2f}"}))

# --- Manual Forecast ---
st.subheader("📅 Forecast by Date")
selected_date = st.date_input("Choose a future date:", datetime.today().date() + timedelta(days=1))
delta_days = (selected_date - data.index[-1].date()).days

if delta_days <= 0:
    st.warning("Please choose a date after today.")
else:
    last_seq = scaled_close[-seq_len:]
    future_seq = []
    for _ in range(delta_days):
        pred = model.predict(last_seq.reshape(1, seq_len, 1))[0][0]
        future_seq.append(pred)
        last_seq = np.append(last_seq[1:], [[pred]], axis=0)

    final_pred = scaler.inverse_transform(np.array(future_seq[-1:]).reshape(-1, 1)).flatten()[0]
    st.success(f"📅 Predicted price for {selected_date.strftime('%Y-%m-%d')}: ₹{final_pred:.2f}")

st.info("⚠️ This BiLSTM-based prediction is for educational purposes only and not financial advice.")

