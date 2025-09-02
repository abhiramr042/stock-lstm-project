
# 📈 Stock Price Forecasting with LSTM

This project implements a Long Short-Term Memory (LSTM) deep learning model to predict and forecast stock prices.  
It uses **Yahoo Finance data**, **TensorFlow/Keras**, and a **Streamlit dashboard** for visualization.

---

## 🚀 Features
- Download stock data dynamically using `yfinance`
- Train an LSTM model on stock closing prices
- Predict stock price trends and compare against actual values
- Forecast future stock prices (N days ahead)
- Interactive Streamlit app with visualizations

---

## 📂 Project Structure
stock-lstm-project/
│── app/
│ └── streamlit_app.py
│── notebooks/
│ └── train_and_save.py
│── models/
│ ├── stock_lstm.keras
│ └── scaler.pkl
│── requirements.txt
│── README.md

▶️ Usage
1. Train the Model
cd notebooks
python train_and_save.py


This saves the model and scaler into models/.

2. Run Streamlit App
cd app
streamlit run streamlit_app.py

📊 Example Output
Blue: Actual stock prices
Red: Predicted test set prices
Green: Forecasted future prices

🛠 Tech Stack
Python
TensorFlow / Keras
Scikit-learn
Streamlit
Yahoo Finance API
Pandas, Numpy, Matplotlib
