import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib


# ---------- METRICS FUNCTION (same as RF/XGB) ----------

def stock_metrics_returns(y_true, y_pred):
    """Full metrics for RETURN prediction"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    mape = np.mean(
        np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 0.01))
    ) * 100

    dir_true = np.sign(np.diff(y_true))
    dir_pred = np.sign(np.diff(y_pred))
    dir_acc = np.mean(dir_true == dir_pred) * 100

    signals = np.where(y_pred > 0, 1, -1)
    strategy_returns = y_true * signals
    sharpe = (
        np.sqrt(252) * np.mean(strategy_returns) / np.std(strategy_returns)
        if np.std(strategy_returns) > 0
        else 0
    )

    naive_rmse = np.sqrt(mean_squared_error(y_true, np.zeros_like(y_true)))
    beats_naive = rmse < naive_rmse

    return {
        "R²": round(r2, 3),
        "RMSE (%)": round(rmse, 3),
        "MAE (%)": round(mae, 3),
        "MAPE (%)": round(mape, 2),
        "Dir.Acc (%)": round(dir_acc, 1),
        "Sharpe Ratio": round(sharpe, 2),
        "Beats Naive": "YES" if beats_naive else "NO",
    }


print("🚀 COMPLETE SVM - Nifty50 Returns Prediction (Model #3)")
print("=" * 65)

# ---------- 1. LOAD + PREPARE DATA (same as other models) ----------

df = pd.read_csv("contents/Nifty50_features_15years.csv", index_col=0, parse_dates=True)

# target = next‑day % return
df["Target_Return"] = df["Close"].pct_change().shift(-1) * 100
df = df.dropna()

split = int(0.8 * len(df))
train_df = df.iloc[:split]
test_df = df.iloc[split:]

features = [
    "RSI_14",
    "SMA_20",
    "SMA_50",
    "MACD",
    "MACD_hist",
    "Stoch_K",
    "ATR_14",
    "Williams_R",
    "ADX_14",
    "CCI_20",
    "ROC_10",
    "Volume_SMA_20",
]

X_train = train_df[features]
X_test = test_df[features]
y_train = train_df["Target_Return"]
y_test = test_df["Target_Return"]

print(f"Dataset: {len(train_df)} train, {len(test_df)} test")
print(
    f"Mean return (train/test): {y_train.mean():.3f}% / {y_test.mean():.3f}%"
)

# ---------- 2. SCALING ----------

scaler_x = StandardScaler()
X_train_s = scaler_x.fit_transform(X_train)
X_test_s = scaler_x.transform(X_test)

scaler_y = StandardScaler()
y_train_s = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_s = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

# ---------- 3. TRAIN SVM ----------

svm_model = SVR(
    kernel="rbf",
    C=1.0,
    epsilon=0.1,
    gamma="scale",
    max_iter=10000,
)

print("Training SVM (may take up to 1–2 minutes)...")
svm_model.fit(X_train_s, y_train_s)
print("✅ SVM trained")

# ---------- 4. PREDICT & INVERSE‑TRANSFORM ----------

y_pred_s = svm_model.predict(X_test_s)
y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()

# ---- BUILD DATE‑WISE ACTUAL & PREDICTED CLOSING PRICE CSV ----

# Actual closing prices from test data
actual_prices = test_df['Close'].to_numpy()
dates = test_df.index

# Reconstruct predicted closing prices from predicted returns (y_pred in %)
N = len(actual_prices)
predicted_prices = np.empty(N, dtype=float)
predicted_prices[0] = actual_prices[0]

for i in range(1, N):
    r = y_pred[i-1] / 100.0          # predicted return for previous day
    predicted_prices[i] = predicted_prices[i-1] * (1.0 + r)

# Create DataFrame with Date, Actual, Predicted
price_df = pd.DataFrame({
    'Date': dates,
    'Actual_Close': actual_prices,
    'Predicted_Close_SVM': predicted_prices
})

# Save to CSV
price_df.to_csv('svm_price_predictions_test.csv', index=False)
print("\n✅ Saved date‑wise prices to svm_price_predictions_test.csv")

# ---------- 5. METRICS ----------

metrics = stock_metrics_returns(y_test, y_pred)

print("\n🏆 SVM RESULTS (Daily Returns):")
print("-" * 35)
for metric, value in metrics.items():
    print(f"{metric:<12}: {value}")

# ---------- 6. PLOT 1 – RETURNS LINE PLOT ----------

plt.figure(figsize=(15, 8))
plt.plot(
    test_df.index,
    y_test.values,
    label="Actual Returns",
    alpha=0.8,
    linewidth=1.5,
    color="#1f77b4",
)
plt.plot(
    test_df.index,
    y_pred,
    label="SVM Predicted Returns",
    alpha=0.8,
    linewidth=1.5,
    color="#d62728",
)
plt.title(
    "SVM: Actual vs Predicted Daily Returns\nNifty50 Test Period",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Date")
plt.ylabel("Daily Return (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("svm_returns.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------- 7. PLOT 2 – CUMULATIVE RETURNS (STRATEGY vs BUY & HOLD) ----------

actual_cum = (1 + y_test / 100).cumprod()
svm_cum = (1 + y_pred / 100).cumprod()

plt.figure(figsize=(15, 8))
plt.plot(
    test_df.index,
    actual_cum,
    label="Buy & Hold",
    linewidth=2.5,
    color="green",
)
plt.plot(
    test_df.index,
    svm_cum,
    label="SVM Strategy",
    linewidth=2.5,
    color="#d62728",
)
plt.title(
    "SVM Trading Strategy vs Buy & Hold\nCumulative Returns (Test Period)",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("svm_cumulative.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------- 8. PLOT 3 – “PRICE‑STYLE” PATH FROM RETURNS (OPTIONAL) ----------

actual_prices = test_df["Close"].to_numpy()
N = len(actual_prices)
predicted_prices = np.empty(N, dtype=float)
predicted_prices[0] = actual_prices[0]

for i in range(1, N):
    r = y_pred[i - 1] / 100.0
    predicted_prices[i] = predicted_prices[i - 1] * (1.0 + r)

plt.figure(figsize=(15, 8))
plt.plot(
    test_df.index,
    actual_prices,
    label="Actual Price",
    linewidth=2.5,
    color="green",
)
plt.plot(
    test_df.index,
    predicted_prices,
    label="SVM Predicted Price",
    linewidth=2.0,
    color="#d62728",
)
plt.title(
    "SVM: Actual vs Predicted Closing Prices\n(derived from predicted returns)",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Date")
plt.ylabel("Closing Price (₹)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("svm_prices.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------- 9. PLOT 4 – PERMUTATION FEATURE IMPORTANCE ----------

from sklearn.inspection import permutation_importance

perm = permutation_importance(
    svm_model, X_test_s, y_test_s, n_repeats=10, random_state=42
)
importance_df = pd.DataFrame(
    {"Feature": features, "Importance": perm.importances_mean}
).sort_values("Importance", ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(importance_df["Feature"], importance_df["Importance"], color="#d62728")
plt.title(
    "SVM Feature Importance (Permutation)\nNifty50 Returns Prediction",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("svm_importance.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------- 10. THESIS TABLE + SAVE ----------

print("\n📊 THESIS TABLE (SVM):")
print("| Metric        | Value    |")
print("|---------------|----------|")
for k, v in metrics.items():
    print(f"| {k:<13} | {v:>8} |")

joblib.dump(svm_model, "svm_nifty50_model.pkl")
joblib.dump({"scaler_x": scaler_x, "scaler_y": scaler_y}, "svm_scalers.pkl")
pd.DataFrame([metrics]).to_csv("svm_results.csv", index=False)
importance_df.to_csv("svm_feature_importance.csv", index=False)

print("\n✅ SVM COMPLETE! Files saved:")
print("- svm_nifty50_model.pkl")
print("- svm_results.csv")
print("- svm_returns.png, svm_cumulative.png, svm_prices.png, svm_importance.png")
