import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Define ticker and 15-year period

#-------------RELIANCE INDUSTRIES----------------
ticker = "RELIANCE.NS"
end_date = datetime.now()
start_date = end_date - timedelta(days=15*365)

# Download data
print("Downloading 15 years of RELIANCE data...")
nifty50 = yf.download(ticker, start=start_date, end=end_date, interval="1d")

# Save to CSV
nifty50.to_csv("contents/RELIANCE_15years_daily.csv")
print(f"Dataset saved: RELIANCE_15years_daily.csv ({len(nifty50)} rows)")

# Display first few rows
print("\nFirst 5 rows:")
print(nifty50.head())

# Display last few rows
print("\nLast 5 rows:")
print(nifty50.tail())
"""
#-------------Nifty50----------------
ticker = "^NSEI"
end_date = datetime.now()
start_date = end_date - timedelta(days=15*365)

# Download data
print("Downloading 15 years of Nifty 50 data...")
nifty50 = yf.download(ticker, start=start_date, end=end_date, interval="1d")

# Save to CSV
nifty50.to_csv("contents/Nifty50_15years_daily.csv")
print(f"Dataset saved: Nifty50_15years_daily.csv ({len(nifty50)} rows)")

# Display first few rows
print("\nFirst 5 rows:")
print(nifty50.head())

# Display last few rows
print("\nLast 5 rows:")
print(nifty50.tail())

#-------------BankNifty----------------
ticker = "^NSEBANK"
end_date = datetime.now()
start_date = end_date - timedelta(days=15*365)

# Download data
print("Downloading 15 years of BankNifty data...")
banknifty = yf.download(ticker, start=start_date, end=end_date, interval="1d")

# Save to CSV
banknifty.to_csv("contents/BankNifty_15years_daily.csv")
print(f"Dataset saved: BankNifty_15years_daily.csv ({len(banknifty)} rows)")

# Display first few rows
print("\nFirst 5 rows:")
print(banknifty.head())

# Display last few rows
print("\nLast 5 rows:")
print(banknifty.tail())

#-------------NIFTY AUTO----------------
ticker = "^CNXAUTO"
end_date = datetime.now()
start_date = end_date - timedelta(days=15*365)

# Download data
print("Downloading 15 years of NIFTYAUTO  data...")
niftyauto = yf.download(ticker, start=start_date, end=end_date, interval="1d")

# Save to CSV
niftyauto.to_csv("contents/NIFTYAUTO_15years_daily.csv")
print(f"Dataset saved: NIFTYAUTO_15years_daily.csv ({len(niftyauto)} rows)")

# Display first few rows
print("\nFirst 5 rows:")
print(niftyauto.head())

# Display last few rows
print("\nLast 5 rows:")
print(niftyauto.tail())

#-------------NIFTY PHARMA----------------
ticker = "^CNXPHARMA"
end_date = datetime.now()
start_date = end_date - timedelta(days=15*365)

# Download data
print("Downloading 15 years of NIFTYPHARMA  data...")
niftypharma = yf.download(ticker, start=start_date, end=end_date, interval="1d")

# Save to CSV
niftypharma.to_csv("contents/NIFTYPHARMA_15years_daily.csv")
print(f"Dataset saved: NIFTYPHARMA_15years_daily.csv ({len(niftypharma)} rows)")

# Display first few rows
print("\nFirst 5 rows:")
print(niftypharma.head())

# Display last few rows
print("\nLast 5 rows:")
print(niftypharma.tail())

#-------------NIFTY IT----------------
ticker = "^CNXIT"
end_date = datetime.now()
start_date = end_date - timedelta(days=15*365)

# Download data
print("Downloading 15 years of NIFTY IT  data...")
niftyit = yf.download(ticker, start=start_date, end=end_date, interval="1d")

# Save to CSV
niftyit.to_csv("contents/NIFTYIT_15years_daily.csv")
print(f"Dataset saved: NIFTYIT_15years_daily.csv ({len(niftyit)} rows)")

# Display first few rows
print("\nFirst 5 rows:")
print(niftyit.head())

# Display last few rows
print("\nLast 5 rows:")
print(niftyit.tail())

#-------------NIFTY METAL----------------
ticker = "^CNXMETAL"
end_date = datetime.now()
start_date = end_date - timedelta(days=15*365)

# Download data
print("Downloading 15 years of NIFTYMETAL  data...")
niftymetal = yf.download(ticker, start=start_date, end=end_date, interval="1d")

# Save to CSV
niftymetal.to_csv("contents/NIFTYMETAL_15years_daily.csv")
print(f"Dataset saved: NIFTYMETAL_15years_daily.csv ({len(niftymetal)} rows)")

# Display first few rows
print("\nFirst 5 rows:")
print(niftymetal.head())

# Display last few rows
print("\nLast 5 rows:")
print(niftymetal.tail())

"""