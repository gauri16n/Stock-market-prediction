import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt

# Title
st.title("📈 Stock Market Prediction App")
st.write("Predict if the S&P 500 index will go up tomorrow using historical data.")

# Load Data
@st.cache_data
def load_data():
    sp500 = yf.Ticker("^GSPC")
    data = sp500.history(period="max")
    data = data[["Open", "High", "Low", "Close", "Volume"]].copy()
    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
    return data

df = load_data()
st.subheader("Sample Data")
st.dataframe(df.tail())

# Feature Engineering
horizons = [2, 5, 60, 250, 1000]
for horizon in horizons:
    rolling_avg = df["Close"].rolling(horizon).mean()
    df[f"Close_Ratio_{horizon}"] = df["Close"] / rolling_avg
    df[f"Trend_{horizon}"] = df["Target"].shift(1).rolling(horizon).sum()

df = df.dropna()
predictors = [col for col in df.columns if col.startswith("Close_Ratio_") or col.startswith("Trend_")]

# Train/Test Split
train = df.iloc[:-100]
test = df.iloc[-100:]

# Model Training
model = LogisticRegression(max_iter=1000)
model.fit(train[predictors], train["Target"])

# Predictions
preds = model.predict(test[predictors])
test["Predictions"] = preds

# Display precision
precision = precision_score(test["Target"], test["Predictions"])
st.write(f"🔍 **Model Precision:** {precision:.2f}")

# Show prediction distribution
st.subheader("Prediction Results")
st.bar_chart(test["Predictions"].value_counts())

# Plot stock closing prices
st.subheader("S&P 500 Closing Price")
st.line_chart(df["Close"])

# Allow user to download predictions
csv = test[["Target", "Predictions"]].to_csv(index=False)
st.download_button("Download Predictions CSV", csv, "predictions.csv", "text/csv")
