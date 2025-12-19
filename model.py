import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load & create target (predict next close)
df = pd.read_csv('contents/Nifty50_features_15years.csv', index_col=0, parse_dates=True)
df['Target'] = df['Close'].shift(-1)  # Next day's close
df = df.dropna()  # Drop last row (no target)

# Features (exclude Target & original prices for cleaner model)
features = ['RSI_14', 'SMA_20', 'SMA_50', 'MACD', 'MACD_hist', 'BB_percentB', 
            'Stoch_K', 'ATR_14', 'Williams_R', 'ADX_14', 'CCI_20', 'ROC_10', 'Volume_SMA_20']

X = df[features]
y = df['Target']

# Time-ordered 80/20 split (CRITICAL: no shuffle for time series)
split = int(0.8 * len(X))
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# 1. MinMaxScaler for LSTM (0-1 range)
mm_scaler = MinMaxScaler()
X_train_mm = mm_scaler.fit_transform(X_train)
X_test_mm = mm_scaler.transform(X_test)

# Convert back to DataFrame for CSV
X_train_mm_df = pd.DataFrame(X_train_mm, columns=features, index=X_train.index)
X_test_mm_df = pd.DataFrame(X_test_mm, columns=features, index=X_test.index)

X_train_mm_df.to_csv('contents/Nifty50_train_mm_scaled.csv')
X_test_mm_df.to_csv('contents/Nifty50_test_mm_scaled.csv')

# 2. StandardScaler for Random Forest
std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_test_std = std_scaler.transform(X_test)

X_train_std_df = pd.DataFrame(X_train_std, columns=features, index=X_train.index)
X_test_std_df = pd.DataFrame(X_test_std, columns=features, index=X_test.index)

X_train_std_df.to_csv('contents/Nifty50_train_std_scaled.csv')
X_test_std_df.to_csv('contents/Nifty50_test_std_scaled.csv')

print("Scaling complete! Ready for model training.")
print("Use X_train_std, X_test_std for Random Forest")
print("Use X_train_mm, X_test_mm for LSTM")
