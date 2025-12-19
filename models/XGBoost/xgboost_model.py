import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from xgboost import XGBRegressor  # ✅ ONLY THIS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

def stock_metrics_returns(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 0.01))) * 100
    
    dir_true = np.sign(np.diff(y_true))
    dir_pred = np.sign(np.diff(y_pred))
    dir_acc = np.mean(dir_true == dir_pred) * 100
    
    signals = np.where(y_pred > 0, 1, -1)
    strategy_returns = y_true * signals
    sharpe = np.sqrt(252) * np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) > 0 else 0
    
    naive_rmse = np.sqrt(mean_squared_error(y_true, np.zeros_like(y_true)))
    beats_naive = rmse < naive_rmse
    
    return {
        'R²': round(r2, 3),
        'RMSE (%)': round(rmse, 3),
        'MAE (%)': round(mae, 3),
        'MAPE (%)': round(mape, 2),
        'Dir.Acc (%)': round(dir_acc, 1),
        'Sharpe Ratio': round(sharpe, 2),
        'Beats Naive': 'YES' if beats_naive else 'NO'
    }

print("🚀 COMPLETE XGBOOST - Nifty50 Returns Prediction")
print("=" * 60)

# Load data (SAME as RF)
df = pd.read_csv('contents/Nifty50_features_15years.csv', index_col=0, parse_dates=True)
df['Target_Return'] = df['Close'].pct_change().shift(-1) * 100
df = df.dropna()

split = int(0.8 * len(df))
train_df = df.iloc[:split]
test_df = df.iloc[split:]

features = ['RSI_14', 'SMA_20', 'SMA_50', 'MACD', 'MACD_hist', 'Stoch_K', 
            'ATR_14', 'Williams_R', 'ADX_14', 'CCI_20', 'ROC_10', 'Volume_SMA_20']

X_train = train_df[features]
X_test = test_df[features]
y_train = train_df['Target_Return']
y_test = test_df['Target_Return']

# Scale
scaler_x = StandardScaler()
X_train_s = scaler_x.fit_transform(X_train)
X_test_s = scaler_x.transform(X_test)

scaler_y = StandardScaler()
y_train_s = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_s = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

# Train XGBoost
xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
xgb.fit(X_train_s, y_train_s)

# Predict
y_pred_s = xgb.predict(X_test_s)
y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()

# Metrics
metrics = stock_metrics_returns(y_test, y_pred)

print("\n🏆 XGBOOST RESULTS:")
for metric, value in metrics.items():
    print(f"{metric:<12}: {value}")

# 4 PLOTS (same as RF)
plt.figure(figsize=(15, 8))
plt.plot(test_df.index, y_test.values, label='Actual Returns', alpha=0.8, linewidth=1.5, color='#1f77b4')
plt.plot(test_df.index, y_pred, label='XGB Predicted', alpha=0.8, linewidth=1.5, color='#2ca02c')
plt.title('XGBoost: Actual vs Predicted Returns', fontsize=14, fontweight='bold')
plt.ylabel('Daily Return (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('models/XGBoost/xgb_returns.png', dpi=300, bbox_inches='tight')
plt.show()

# Price plot
# After you have y_test (actual returns, in %) and y_pred (predicted returns, in %)
actual_prices = test_df['Close'].to_numpy()      # shape (N,)
pred_returns  = np.asarray(y_pred, dtype=float)  # shape (N,)

print("Shapes:", actual_prices.shape, pred_returns.shape)
print("Sample returns:", pred_returns[:10])

# 1) Make sure lengths match
N = len(actual_prices)
if len(pred_returns) != N:
    # If returns are one shorter (common), align explicitly:
    pred_returns = pred_returns[:N]

# 2) Build predicted price path from FIRST actual price
predicted_prices = np.empty(N, dtype=float)
predicted_prices[0] = actual_prices[0]

for i in range(1, N):
    r = pred_returns[i-1] / 100.0          # percent → fraction
    predicted_prices[i] = predicted_prices[i-1] * (1.0 + r)

print("Pred price sample:", predicted_prices[:10])

# 3) Plot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.figure(figsize=(15, 8))
plt.plot(test_df.index, actual_prices,
         label='Actual Price', linewidth=2.5, color='green')
plt.plot(test_df.index, predicted_prices,
         label='XGB Predicted Price', linewidth=2.0, color='red')
plt.title('XGBoost: Actual vs Predicted Closing Prices')
plt.xlabel('Date')
plt.ylabel('Closing Price (₹)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('models/XGBoost/xgb_prices_fixed.png', dpi=300, bbox_inches='tight')
plt.show()


# Save
joblib.dump(xgb, 'models/XGBoost/xgb_model.pkl')
pd.DataFrame([metrics]).to_csv('models/XGBoost/xgb_results.csv', index=False)
print("\n✅ XGBOOST COMPLETE! 2 plots + model saved!")
