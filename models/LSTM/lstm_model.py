import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib

# ================= METRICS FUNCTION ================= #

def stock_metrics_price(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Direction of price change
    dir_true = np.sign(np.diff(y_true))
    dir_pred = np.sign(np.diff(y_pred))
    dir_acc = np.mean(dir_true == dir_pred) * 100

    return {
        "R²": round(r2, 3),
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "MAPE (%)": round(mape, 2),
        "Dir.Acc (%)": round(dir_acc, 1),
    }

# ================= DATA LOADING ================= #

print("🚀 LSTM – Nifty50 Closing Price Prediction")
print("=" * 60)

df = pd.read_csv("contents/Nifty50_features_15years.csv", index_col=0, parse_dates=True)

# Use only Close price for LSTM (sequence of prices)
data = df[["Close"]].copy()

# Train / test split by time (same 80/20 as other models)
split = int(0.8 * len(data))
train_data = data.iloc[:split]
test_data = data.iloc[split:]

print(f"Train points: {len(train_data)}, Test points: {len(test_data)}")

# ================= SCALING ================= #

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# ================= SEQUENCE CREATION ================= #

def create_sequences(series, time_steps=60):
    X, y = [], []
    for i in range(time_steps, len(series)):
        X.append(series[i - time_steps:i, 0])
        y.append(series[i, 0])
    return np.array(X), np.array(y)

TIME_STEPS = 30  # Reduced from 60

# Recreate sequences with smaller window
X_train, y_train = create_sequences(train_scaled, TIME_STEPS)
X_test, y_test_seq = create_sequences(test_scaled, TIME_STEPS)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# Align actual prices for test set (skip first TIME_STEPS days)
test_index = test_data.index[TIME_STEPS:]
actual_test_prices = test_data["Close"].iloc[TIME_STEPS:].to_numpy()

model = Sequential([
    LSTM(32, return_sequences=False, input_shape=(TIME_STEPS, 1)),  # Single layer
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

print("Training LSTM (this may take a few minutes)...")
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1,
)
print("✅ LSTM trained")

# ================= PREDICTION ================= #

y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled).ravel()

# ================= METRICS ================= #

metrics = stock_metrics_price(actual_test_prices, y_pred)

print("\n🏆 LSTM RESULTS (Closing Price):")
print("-" * 35)
for k, v in metrics.items():
    print(f"{k:<12}: {v}")

# ================= CSV: DATE, ACTUAL, PREDICTED ================= #
model = LSTM
results_df = pd.DataFrame(
    {
        "Date": test_index,
        "Actual_Close": actual_test_prices,
        "Predicted_Close_LSTM": y_pred,
    }
)
results_df.to_csv("models/LSTM/lstm_price_predictions_test.csv", index=False)
print("\n✅ Saved date‑wise prices to lstm_price_predictions_test.csv")

# ================= PLOTS ================= #

# 1) Actual vs Predicted prices (test)
plt.figure(figsize=(15, 8))
plt.plot(test_index, actual_test_prices, label="Actual Price", color="green")
plt.plot(test_index, y_pred, label="LSTM Predicted Price", color="red")
plt.title("LSTM: Actual vs Predicted Closing Prices – Nifty50 (Test)")
plt.xlabel("Date")
plt.ylabel("Closing Price (₹)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("lstm_prices.png", dpi=300, bbox_inches="tight")
plt.show()

# 2) Zoomed last 200 points
plt.figure(figsize=(15, 8))
plt.plot(
    test_index[-200:],
    actual_test_prices[-200:],
    label="Actual Price",
    color="green",
)
plt.plot(
    test_index[-200:],
    y_pred[-200:],
    label="LSTM Predicted Price",
    color="red",
)
plt.title("LSTM: Actual vs Predicted (Last 200 Days)")
plt.xlabel("Date")
plt.ylabel("Closing Price (₹)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("models/LSTM/lstm_prices_last200.png", dpi=300, bbox_inches="tight")
plt.show()

# 3) Training loss curve
plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("LSTM Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("models/LSTM/lstm_loss.png", dpi=300, bbox_inches="tight")
plt.show()

# ================= SAVE MODEL & SCALER ================= #

model.save("models/LSTM/lstm_nifty50_model.h5")
joblib.dump(scaler, "models/LSTM/lstm_close_scaler.pkl")

print("\n✅ LSTM COMPLETE! Files saved:")
print("- lstm_nifty50_model.h5")
print("- lstm_close_scaler.pkl")
print("- lstm_price_predictions_test.csv")
print("- lstm_prices.png, lstm_prices_last200.png, lstm_loss.png")