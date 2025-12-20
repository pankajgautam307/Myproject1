import pandas as pd
import numpy as np
import talib
import warnings
warnings.filterwarnings('ignore')

# -----------------Load your RELIANCE data------------------------------------------------------

df = pd.read_csv('contents/RELIANCE_15years_daily.csv', index_col=0, parse_dates=True)
df = df.sort_index()  # Ensure chronological order
print(f"Original columns: {list(df.columns)}")
print(f"Original: {len(df)} rows")

def compute_correct_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Wilder's RSI - vectorized (SIMPLE & CORRECT)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_features(df):
    # Compute all 15 features using pure Pandas/NumPy
    
    # 1-5. Lagged features (Close lag1-5, Open lag1-5, High lag1-5, Low lag1-5, Volume lag1)
    for lag in range(1, 6):
        df[f'Close_lag{lag}'] = df['Close'].shift(lag)
    
    for lag in range(1, 6):
        df[f'Open_lag{lag}'] = df['Open'].shift(lag)
    
    for lag in range(1, 6):
        df[f'High_lag{lag}'] = df['High'].shift(lag)
    
    for lag in range(1, 6):
        df[f'Low_lag{lag}'] = df['Low'].shift(lag)
    
    df['Volume_lag1'] = df['Volume'].shift(1)
    
    # 6. RSI (14-period)
    df['RSI_14'] = compute_correct_rsi(df['Close'])
    
    # 7-9. SMA (9, 20, 50)
    df['SMA_9'] = df['Close'].rolling(9).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    
    # 10-12. MACD (12,26,9)
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # 13. Bollinger Bands %B (20,2)
    bb_middle = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_upper'] = bb_middle + (bb_std * 2)
    df['BB_lower'] = bb_middle - (bb_std * 2)
    df['BB_percentB'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # 14. Stochastic %K (14,3,3)
    low_min = df['Low'].rolling(14).min()
    high_max = df['High'].rolling(14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    
    # 15. ATR (14)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    df['ATR_14'] = tr.rolling(14).mean()
    
    # 16. Williams %R (14)
    high_max14 = df['High'].rolling(14).max()
    low_min14 = df['Low'].rolling(14).min()
    df['Williams_R'] = -100 * ((high_max14 - df['Close']) / (high_max14 - low_min14))
    
    # 17. ADX (simplified trend strength)
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff() * -1
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_di = 100 * (plus_dm.rolling(14).mean() / df['ATR_14'])
    minus_di = 100 * (minus_dm.rolling(14).mean() / df['ATR_14'])
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    df['ADX_14'] = dx.rolling(14).mean()
    
    # 18. CCI (20)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    mean_tp = tp.rolling(20).mean()
    # Manual mean absolute deviation (stable)
    mean_dev = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
    df['CCI_20'] = (tp - mean_tp) / (0.015 * mean_dev)
    
    # 19. ROC (10)
    df['ROC_10'] = df['Close'].pct_change(10) * 100
    
    # 20. Volume SMA (5)
    df['Volume_SMA_5'] = df['Volume'].rolling(5).mean()
    
    return df

# Compute features
df = compute_features(df)

# FIXED: Only select columns that actually exist
available_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
if 'Adj Close' in df.columns:
    available_cols.append('Adj Close')

feature_cols =  available_cols + [
    'Open_lag1', 'High_lag1','Low_lag1','Close_lag1', 'Volume_lag1', 'RSI_14', 
    'SMA_9', 'SMA_20', 'SMA_50', 'MACD', 'MACD_hist', 'BB_percentB', 'Stoch_K', 
    'ATR_14','Williams_R', 'ADX_14', 'CCI_20', 'ROC_10', 'Volume_SMA_5'
]

# Filter to only existing columns
existing_features = [col for col in feature_cols if col in df.columns]
df_features = df[existing_features].dropna()

print(f" Clean dataset: {len(df_features)} rows with {len(existing_features)} features")
print(f"Features: {existing_features}")

# Save
df_features.to_csv('contents/RELIANCE_features_15years.csv')
print(" Saved to 'RELIANCE_features_15years.csv'")
print("\nFirst 5 rows:")
print(df_features.head())