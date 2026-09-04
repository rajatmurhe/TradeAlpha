import os
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "HINDUNILVR.NS", "SBIN.NS",
    "BHARTIARTL.NS", "MARUTI.NS", "SUNPHARMA.NS", "TATASTEEL.NS", "ULTRACEMCO.NS"
]

MODEL_PATH = os.path.join("models", "xgboost_forecaster.pkl")
OUTPUT_FILE = os.path.join("data", "processed", "latest_forecast.csv")

# Exact 20 features expected by the trained XGBoost model
FEATURES = [
    "return_1d", "return_5d", "return_10d", "return_20d",
    "sma_5", "sma_20", "sma_50", "price_vs_sma5", "price_vs_sma20", "price_vs_sma50",
    "momentum_5", "momentum_10", "volatility_5", "volatility_20", "volatility_60",
    "high_low_range", "open_close_range", "volume_change", "volume_sma_20", "volume_ratio"
]

def fetch_and_engineer_live_data():
    """Fetches the last 100 days of data to fulfill the 60-day rolling window requirements."""
    all_data = []
    
    for ticker in TICKERS:
        try:
            # Download individually to safely avoid pandas MultiIndex conflicts
            df = yf.download(ticker, period="100d", progress=False)
            if df.empty:
                continue
                
            df = df.reset_index()
            df['Ticker'] = ticker
            
            # Flatten multi-level columns if yfinance returns them (newer yfinance versions)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
                
            all_data.append(df)
        except Exception as e:
            print(f"Failed to fetch {ticker}: {e}")
            
    df = pd.concat(all_data, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    
    # --------------------------------------------------------
    # LIVE FEATURE ENGINEERING
    # --------------------------------------------------------
    # Returns
    df["return_1d"] = df.groupby("Ticker")["Close"].pct_change(1)
    df["return_5d"] = df.groupby("Ticker")["Close"].pct_change(5)
    df["return_10d"] = df.groupby("Ticker")["Close"].pct_change(10)
    df["return_20d"] = df.groupby("Ticker")["Close"].pct_change(20)

    # Moving Averages
    df["sma_5"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(5, min_periods=5).mean())
    df["sma_20"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(20, min_periods=20).mean())
    df["sma_50"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(50, min_periods=50).mean())

    # Price vs SMA
    df["price_vs_sma5"] = df["Close"] / df["sma_5"] - 1
    df["price_vs_sma20"] = df["Close"] / df["sma_20"] - 1
    df["price_vs_sma50"] = df["Close"] / df["sma_50"] - 1

    # Momentum
    df["momentum_5"] = df["Close"] / df.groupby("Ticker")["Close"].shift(5) - 1
    df["momentum_10"] = df["Close"] / df.groupby("Ticker")["Close"].shift(10) - 1

    # Volatility
    df["volatility_5"] = df.groupby("Ticker")["return_1d"].transform(lambda x: x.rolling(5, min_periods=5).std())
    df["volatility_20"] = df.groupby("Ticker")["return_1d"].transform(lambda x: x.rolling(20, min_periods=20).std())
    df["volatility_60"] = df.groupby("Ticker")["return_1d"].transform(lambda x: x.rolling(60, min_periods=60).std())

    # Ranges
    df["high_low_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["open_close_range"] = (df["Close"] - df["Open"]) / df["Open"]

    # Volume
    df["volume_change"] = df.groupby("Ticker")["Volume"].pct_change()
    df["volume_sma_20"] = df.groupby("Ticker")["Volume"].transform(lambda x: x.rolling(20, min_periods=20).mean())
    df["volume_ratio"] = df["Volume"] / df["volume_sma_20"]

    # Isolate only the latest trading day for each ticker for LIVE inference
    latest_df = df.groupby("Ticker").tail(1).copy()
    
    # Drop rows that somehow still have NaNs (e.g. IPOs with <60 days of data)
    latest_df = latest_df.dropna(subset=FEATURES).reset_index(drop=True)
    return latest_df

def run_live_inference():
    """Generates forecasts using the trained XGBoost model and updates the CSV."""
    print("Fetching live market data and engineering features...")
    live_data = fetch_and_engineer_live_data()
    
    if live_data.empty:
        raise ValueError("Live data fetch failed or returned empty.")

    print("Loading XGBoost forecaster...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model missing at {MODEL_PATH}")
        
    model = joblib.load(MODEL_PATH)
    
    print("Running inference...")
    X = live_data[FEATURES]
    predictions = model.predict(X)
    
    # Format the output dataframe expected by the UI and Future Engine
    live_data["predicted_return"] = predictions
    live_data["predicted_price"] = live_data["Close"] * (1 + live_data["predicted_return"])
    
    # Establish direction (0.10% Hurdle)
    hurdle_rate = 0.0010
    live_data["predicted_direction"] = np.where(
        live_data["predicted_return"] > hurdle_rate, "BULLISH",
        np.where(live_data["predicted_return"] < -hurdle_rate, "BEARISH", "NEUTRAL")
    )
    
    # Select final columns and save
    final_cols = ["Date", "Ticker", "Close", "volatility_20", "momentum_10", 
                  "predicted_return", "predicted_price", "predicted_direction"]
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    live_data[final_cols].to_csv(OUTPUT_FILE, index=False)
    
    print(f"Successfully generated new forecasts for {len(live_data)} assets.")
    print(f"Data saved to {OUTPUT_FILE}")
    return True

if __name__ == "__main__":
    run_live_inference()