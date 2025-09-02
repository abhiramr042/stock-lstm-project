import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="📈 Stock Price LSTM Forecaster", layout="wide")

# --------------------------
# Sidebar
# --------------------------
st.sidebar.header("Stock Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
start_date = st.sidebar.text_input("Start Date", "2015-01-01")
end_date = st.sidebar.text_input("End Date", datetime.today().strftime("%Y-%m-%d"))
seq_length = st.sidebar.number_input("Sequence Length", min_value=30, max_value=200, value=60, step=1)
days_to_forecast = st.sidebar.number_input("Days to Forecast", min_value=1, max_value=90, value=21, step=1)

# --------------------------
# Load Stock Data
# --------------------------
st.title(f"{ticker} Stock Forecast")

df = yf.download(ticker, start=start_date, end=end_date)

if df.empty:
    st.error("⚠️ No data found for this ticker/date range. Try different inputs.")
    st.stop()

st.line_chart(df["Close"], use_container_width=True)
st.write(f"Showing data for **{ticker}** from {start_date} to {end_date}")

# --------------------------
# Scaling (Close only)
# --------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close = scaler.fit_transform(df[["Close"]])  # only Close

# --------------------------
# Prepare Sequences
# --------------------------
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 0])  # Close price
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_close, seq_length)

if len(X) == 0:
    st.error("⚠️ Not enough data after feature engineering. Try a longer date range.")
    st.stop()

split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# --------------------------
# Build LSTM Model
# --------------------------
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

# --------------------------
# Predictions
# --------------------------
predicted = model.predict(X_test)

predicted_prices = scaler.inverse_transform(predicted)[:, 0]
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))[:, 0]

# Plot Predictions
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index[-len(actual_prices):], actual_prices, label="Actual Close", color="blue")
ax.plot(df.index[-len(predicted_prices):], predicted_prices, label="Predicted Close", color="red")
ax.set_title(f"{ticker} Stock Price Prediction")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
st.pyplot(fig)

# --------------------------
# Forecast Future
# --------------------------
last_sequence = scaled_close[-seq_length:]
future = []

current_seq = last_sequence
for _ in range(days_to_forecast):
    pred = model.predict(current_seq.reshape(1, seq_length, 1))
    future.append(pred[0, 0])
    current_seq = np.vstack((current_seq[1:], pred))

future_prices = scaler.inverse_transform(np.array(future).reshape(-1, 1))[:, 0]
future_dates = pd.date_range(df.index[-1] + timedelta(days=1), periods=days_to_forecast)

forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Close": future_prices})

# Plot Historical + Forecast
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(df.index, df["Close"], label="Historical Close", color="blue")
ax2.plot(forecast_df["Date"], forecast_df["Predicted Close"], label="Forecasted Close", color="green")
ax2.set_title(f"{ticker} Stock Forecast ({days_to_forecast} days)")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price (USD)")
ax2.legend()
st.pyplot(fig2)

# Show forecasted values in table
st.subheader("Forecasted Prices")
st.dataframe(forecast_df)
