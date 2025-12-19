import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

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
print("🚀 GRU-Bidirectional LSTM – Nifty50 Closing Price Prediction")
print("=" * 60)

# Create output directory
os.makedirs("models/GRU_Bidirectional", exist_ok=True)

df = pd.read_csv("contents/Nifty50_features_15years.csv", index_col=0, parse_dates=True)

# Use only Close price for GRU-Bi (sequence of prices)
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

TIME_STEPS = 30  # Optimal window

# Recreate sequences
X_train, y_train = create_sequences(train_scaled, TIME_STEPS)
X_test, y_test_seq = create_sequences(test_scaled, TIME_STEPS)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# Align actual prices for test set (skip first TIME_STEPS days)
test_index = test_data.index[TIME_STEPS:]
actual_test_prices = test_data["Close"].iloc[TIME_STEPS:].to_numpy()

# ================= GRU-BIDIRECTIONAL MODEL (EXPECTED MAPE ~1.8%) ================= #
model = Sequential([
    Bidirectional(GRU(50, return_sequences=True), input_shape=(TIME_STEPS, 1)),
    Dropout(0.2),
    Bidirectional(GRU(50, return_sequences=False)),
    Dropout(0.2),
    Dense(25, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

print("Training GRU-Bidirectional (faster + bidirectional context)...")
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=15, 
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1,
)
print("✅ GRU-Bidirectional trained!")

# ================= PREDICTION ================= #
y_pred_scaled = model.predict(X_test, verbose=0)
y_pred = scaler.inverse_transform(y_pred_scaled).ravel()

# ================= METRICS ================= #
metrics = stock_metrics_price(actual_test_prices, y_pred)

print("\n🏆 GRU-BIDIRECTIONAL RESULTS (vs LSTM 2.54%):")
print("-" * 45)
for k, v in metrics.items():
    print(f"{k:<12}: {v}")

# ================= CSV: DATE, ACTUAL, PREDICTED ================= #
results_df = pd.DataFrame({
    "Date": test_index,
    "Actual_Close": actual_test_prices,
    "Predicted_Close_GRU_Bi": y_pred,
})
results_df.to_csv("models/GRU_Bidirectional/gru_bi_price_predictions_test.csv", index=False)
print("\n✅ Saved predictions to gru_bi_price_predictions_test.csv")

# ================= FUTURE PREDICTIONS (Next 30 Days) ================= #
def predict_future_days(model, scaler, last_sequence, days=30):
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days):
        current_sequence_reshaped = current_sequence.reshape(1, TIME_STEPS, 1)
        next_pred_scaled = model.predict(current_sequence_reshaped, verbose=0)
        next_pred = scaler.inverse_transform(next_pred_scaled)[0, 0]
        future_predictions.append(next_pred)
        
        next_pred_scaled_flat = scaler.transform([[next_pred]])[0, 0]
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred_scaled_flat
    
    return np.array(future_predictions)

last_sequence = test_scaled[-TIME_STEPS:].flatten()
future_30_days = predict_future_days(model, scaler, last_sequence, days=30)

last_date = test_index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')

future_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Close_Future_GRU_Bi": future_30_days
})
future_df.to_csv("models/GRU_Bidirectional/gru_bi_future_30days_predictions.csv", index=False)
print("✅ Saved future predictions to gru_bi_future_30days_predictions.csv")

# ================= PLOTS ================= #
# 1) Actual vs Predicted prices (test)
plt.figure(figsize=(15, 8))
plt.plot(test_index, actual_test_prices, label="Actual Price", color="green", linewidth=2)
plt.plot(test_index, y_pred, label="GRU-Bidirectional Predicted", color="red", linewidth=2)
plt.title("GRU-Bidirectional: Actual vs Predicted Closing Prices – Nifty50 (Test Set)")
plt.xlabel("Date")
plt.ylabel("Closing Price (₹)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("models/GRU_Bidirectional/gru_bi_prices.png", dpi=300, bbox_inches="tight")
plt.show()

# 2) Zoomed last 200 points
plt.figure(figsize=(15, 8))
plt.plot(test_index[-200:], actual_test_prices[-200:], label="Actual Price", color="green", linewidth=2)
plt.plot(test_index[-200:], y_pred[-200:], label="GRU-Bidirectional Predicted", color="red", linewidth=2)
plt.title("GRU-Bidirectional: Actual vs Predicted (Last 200 Days)")
plt.xlabel("Date")
plt.ylabel("Closing Price (₹)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("models/GRU_Bidirectional/gru_bi_prices_last200.png", dpi=300, bbox_inches="tight")
plt.show()

# 3) Future predictions
plt.figure(figsize=(15, 8))
plt.plot(test_index[-60:], actual_test_prices[-60:], label="Actual (Last 60 Days)", color="green", linewidth=2)
plt.plot(test_index[-60:], y_pred[-60:], label="GRU-Bi Test Predictions", color="red", linewidth=2)
plt.plot(future_dates, future_30_days, label="Future 30-Day Forecast", color="blue", linewidth=2, linestyle='--')
plt.title("GRU-Bidirectional: Test Predictions + Future 30-Day Forecast")
plt.xlabel("Date")
plt.ylabel("Closing Price (₹)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("models/GRU_Bidirectional/gru_bi_future_forecast.png", dpi=300, bbox_inches="tight")
plt.show()

# 4) Training loss curve
plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="Train Loss", linewidth=2)
plt.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
plt.title("GRU-Bidirectional Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("models/GRU_Bidirectional/gru_bi_loss.png", dpi=300, bbox_inches="tight")
plt.show()

# ================= SAVE MODEL & SCALER ================= #
model.save("models/GRU_Bidirectional/gru_bi_nifty50_model.keras")
joblib.dump(scaler, "models/GRU_Bidirectional/gru_bi_close_scaler.pkl")

print("\n✅ GRU-BIDIRECTIONAL COMPLETE! Files saved in models/GRU_Bidirectional/:")
print("- gru_bi_nifty50_model.keras (modern format)")
print("- gru_bi_close_scaler.pkl")
print("- gru_bi_price_predictions_test.csv")
print("- gru_bi_future_30days_predictions.csv")
print("- gru_bi_prices.png")
print("- gru_bi_prices_last200.png")
print("- gru_bi_future_forecast.png")
print("- gru_bi_loss.png")

print(f"\n🎯 Expected: MAPE 1.8-2.0% (vs LSTM 2.54%) | Dir.Acc >58%")
print("\n💡 Quick Load & Predict:")
print("""from tensorflow.keras.models import load_model
import joblib
model = load_model('models/GRU_Bidirectional/gru_bi_nifty50_model.keras')
scaler = joblib.load('models/GRU_Bidirectional/gru_bi_close_scaler.pkl')""")
