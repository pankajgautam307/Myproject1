import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Define S&P 500 ticker and 15-year period
ticker = "^NSEI"
end_date = datetime.now()
start_date = end_date - timedelta(days=15*365)

# Download data
print("Downloading 15 years of Nifty 50 data...")
nifty50 = yf.download(ticker, start=start_date, end=end_date, interval="1d")

# Save to CSV
nifty50.to_csv("contents/Nifty50_15years_daily.csv")
print(f"Dataset saved: sp500_15years_daily.csv ({len(nifty50)} rows)")

# Display first few rows
print("\nFirst 5 rows:")
print(nifty50.head())

# Display last few rows
print("\nLast 5 rows:")
print(nifty50.tail())
