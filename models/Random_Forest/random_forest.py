import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

def stock_metrics_returns(y_true, y_pred):
    """Full metrics for RETURN prediction"""
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

print("🚀 COMPLETE RANDOM FOREST - Nifty50 Returns + PRICE PLOTS")
print("=" * 60)

# 1. LOAD + PREPARE DATA
df = pd.read_csv('contents/Nifty50_features_15years.csv', index_col=0, parse_dates=True)
df['Target_Return'] = df['Close'].pct_change().shift(-1) * 100
df = df.dropna()

split = int(0.8 * len(df))
train_df = df.iloc[:split]
test_df = df.iloc[split:]

# 2. FEATURES
features = ['RSI_14', 'SMA_20', 'SMA_50', 'MACD', 'MACD_hist', 'Stoch_K', 
            'ATR_14', 'Williams_R', 'ADX_14', 'CCI_20', 'ROC_10', 'Volume_SMA_20']

X_train = train_df[features]
X_test = test_df[features]
y_train = train_df['Target_Return']
y_test = test_df['Target_Return']

print(f"Dataset: {len(train_df)} train, {len(test_df)} test")

# 3. SCALE
scaler_x = StandardScaler()
X_train_s = scaler_x.fit_transform(X_train)
X_test_s = scaler_x.transform(X_test)

scaler_y = StandardScaler()
y_train_s = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_s = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

# 4. TRAIN + PREDICT
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train_s, y_train_s)
y_pred_s = rf.predict(X_test_s)
y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()

# 5. METRICS (YOUR RESULTS)
metrics = stock_metrics_returns(y_test, y_pred)
print("\n🏆 YOUR RESULTS:")
for metric, value in metrics.items():
    print(f"{metric:<12}: {value}")

# 6. PLOT 1: RETURNS (Actual vs Predicted)
plt.figure(figsize=(15, 8))
plt.plot(test_df.index, y_test.values, label='Actual Returns', alpha=0.8, linewidth=1.5, color='#1f77b4')
plt.plot(test_df.index, y_pred, label='RF Predicted Returns', alpha=0.8, linewidth=1.5, color='#ff7f0e')
plt.title('Random Forest: Actual vs Predicted Daily Returns\nNifty50 Test Period', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Daily Return (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('rf_returns.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. PLOT 2: **PRICE PLOT** (Actual Closing vs Predicted Closing)
plt.figure(figsize=(15, 8))

# Actual closing prices
actual_prices = test_df['Close'].values

# Predicted cumulative prices (start from first test price)
predicted_prices = np.cumsum(y_pred) + actual_prices[0]

plt.plot(test_df.index, actual_prices, label='Actual Closing Price', linewidth=2.5, color='green', alpha=0.9)
plt.plot(test_df.index, predicted_prices, label='Predicted Closing Price (RF)', linewidth=2.5, color='red', alpha=0.9)
plt.title('Random Forest: Actual vs Predicted Closing Prices\nNifty50 Test Period', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Closing Price (₹)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('rf_prices.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. PLOT 3: Cumulative Returns Strategy
plt.figure(figsize=(15, 8))
actual_cum = (1 + y_test/100).cumprod()
pred_cum = (1 + y_pred/100).cumprod()
plt.plot(test_df.index, actual_cum, label='Buy & Hold', linewidth=2.5, color='green')
plt.plot(test_df.index, pred_cum, label='RF Strategy', linewidth=2.5, color='orange')
plt.title('RF Trading Strategy vs Buy & Hold\nCumulative Returns', fontsize=14, fontweight='bold')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('rf_cumulative.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. PLOT 4: Feature Importance
plt.figure(figsize=(10, 8))
importance_df = pd.DataFrame({'Feature': features, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=True)
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('rf_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 10. THESIS TABLE
print("\n📊 THESIS TABLE:")
print("| Metric        | Value    |")
print("|---------------|----------|")
for k, v in metrics.items():
    print(f"| {k:<13} | {v:>8} |")

# 11. SAVE
joblib.dump(rf, 'rf_complete_model.pkl')
joblib.dump({'scaler_x': scaler_x, 'scaler_y': scaler_y}, 'scalers.pkl')
pd.DataFrame([metrics]).to_csv('rf_results.csv', index=False)
print("\n✅ 4 PLOTS + MODEL SAVED!")
