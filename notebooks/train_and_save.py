import os
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ---------------------------
# Parameters
# ---------------------------
TICKER = "AAPL"
START_DATE = "2010-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
SEQ_LENGTH = 60
EPOCHS = 20
BATCH_SIZE = 32

# ---------------------------
# Create models directory
# ---------------------------
os.makedirs("models", exist_ok=True)

# ---------------------------
# Download Data
# ---------------------------
print(f"Downloading {TICKER} data...")
df = yf.download(TICKER, start=START_DATE, end=END_DATE)

# ---------------------------
# Feature Engineering
# ---------------------------
df["SMA_14"] = df["Close"].rolling(window=14).mean()

delta = df["Close"].diff().values  # convert to numpy array
gain = np.where(delta > 0, delta, 0).reshape(-1)   # force 1D
loss = np.where(delta < 0, -delta, 0).reshape(-1)  # force 1D

avg_gain = pd.Series(gain).rolling(window=14).mean()
avg_loss = pd.Series(loss).rolling(window=14).mean() 
rs = avg_gain / avg_loss
df["RSI_14"] = 100 - (100 / (1 + rs))

features = df[["Close", "SMA_14", "RSI_14"]].fillna(method="bfill").fillna(method="ffill")

print("Feature sample:")
print(features.head())

# ---------------------------
# Scaling
# ---------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(features)

# Save scaler
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("✅ Scaler saved to models/scaler.pkl")

# ---------------------------
# Create Sequences
# ---------------------------
X, y = [], []
for i in range(SEQ_LENGTH, len(scaled_data)):
    X.append(scaled_data[i-SEQ_LENGTH:i])   # seq_length timesteps
    y.append(scaled_data[i, 0])             # predict "Close" only

X, y = np.array(X), np.array(y)

print(f"X shape: {X.shape}, y shape: {y.shape}")

# ---------------------------
# Build LSTM Model
# ---------------------------
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))  # predict only Close

model.compile(optimizer="adam", loss="mean_squared_error")

# ---------------------------
# Train Model
# ---------------------------
print("Training LSTM...")
model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS)

# ---------------------------
# Save Model
# ---------------------------
model.save("models/stock_lstm.keras")
print("✅ Model saved to models/stock_lstm.keras")
