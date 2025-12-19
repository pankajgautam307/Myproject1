import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
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
print("🚀 LSTM – Nifty50 Closing Price Prediction (Enhanced)")
print("=" * 60)

# Create output directory
os.makedirs("models/LSTM", exist_ok=True)

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

# ================= ENHANCED MODEL ARCHITECTURE ================= #
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(TIME_STEPS, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

print("Training Enhanced LSTM (with EarlyStopping)...")
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=25, 
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train,
    y_train,
    epochs=100,  # Increased epochs
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1,
)
print("✅ Enhanced LSTM trained")

# ================= PREDICTION ================= #
y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled).ravel()

# ================= METRICS ================= #
metrics = stock_metrics_price(actual_test_prices, y_pred)

print("\n🏆 ENHANCED LSTM RESULTS (Closing Price):")
print("-" * 35)
for k, v in metrics.items():
    print(f"{k:<12}: {v}")

# ================= CSV: DATE, ACTUAL, PREDICTED ================= #
results_df = pd.DataFrame(  # ✅ Fixed: removed model = LSTM bug
    {
        "Date": test_index,
        "Actual_Close": actual_test_prices,
        "Predicted_Close_LSTM": y_pred,
    }
)
results_df.to_csv("models/LSTM/lstm_price_predictions_test.csv", index=False)
print("\n✅ Saved date-wise prices to models/LSTM/lstm_price_predictions_test.csv")

# ================= FUTURE PREDICTIONS (Next 30 Days) ================= #
def predict_future_days(model, scaler, last_sequence, days=30):
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days):
        # Reshape for prediction
        current_sequence_reshaped = current_sequence.reshape(1, TIME_STEPS, 1)
        next_pred_scaled = model.predict(current_sequence_reshaped, verbose=0)
        next_pred = scaler.inverse_transform(next_pred_scaled)[0, 0]
        future_predictions.append(next_pred)
        
        # Update sequence: remove first, add new prediction (scaled)
        next_pred_scaled_flat = scaler.transform([[next_pred]])[0, 0]
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred_scaled_flat
    
    return np.array(future_predictions)

# Get last sequence from test set for future predictions
last_sequence = test_scaled[-TIME_STEPS:].flatten()
future_30_days = predict_future_days(model, scaler, last_sequence, days=30)

# Create future dates
last_date = test_index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')

future_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Close_Future": future_30_days
})
future_df.to_csv("models/LSTM/lstm_future_30days_predictions.csv", index=False)
print("✅ Saved future 30-day predictions to models/LSTM/lstm_future_30days_predictions.csv")

# ================= PLOTS ================= #
# 1) Actual vs Predicted prices (test)
plt.figure(figsize=(15, 8))
plt.plot(test_index, actual_test_prices, label="Actual Price", color="green", linewidth=2)
plt.plot(test_index, y_pred, label="Enhanced LSTM Predicted", color="red", linewidth=2)
plt.title("Enhanced LSTM: Actual vs Predicted Closing Prices – Nifty50 (Test Set)")
plt.xlabel("Date")
plt.ylabel("Closing Price (₹)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("models/LSTM/lstm_prices.png", dpi=300, bbox_inches="tight")
plt.show()

# 2) Zoomed last 200 points
plt.figure(figsize=(15, 8))
plt.plot(test_index[-200:], actual_test_prices[-200:], label="Actual Price", color="green", linewidth=2)
plt.plot(test_index[-200:], y_pred[-200:], label="Enhanced LSTM Predicted", color="red", linewidth=2)
plt.title("Enhanced LSTM: Actual vs Predicted (Last 200 Days)")
plt.xlabel("Date")
plt.ylabel("Closing Price (₹)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("models/LSTM/lstm_prices_last200.png", dpi=300, bbox_inches="tight")
plt.show()

# 3) Future predictions
plt.figure(figsize=(15, 8))
plt.plot(test_index[-60:], actual_test_prices[-60:], label="Actual (Last 60 Days)", color="green", linewidth=2)
plt.plot(test_index[-60:], y_pred[-60:], label="LSTM Test Predictions", color="red", linewidth=2)
plt.plot(future_dates, future_30_days, label="Future 30-Day Forecast", color="blue", linewidth=2, linestyle='--')
plt.title("LSTM: Test Predictions + Future 30-Day Forecast")
plt.xlabel("Date")
plt.ylabel("Closing Price (₹)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("models/LSTM/lstm_future_forecast.png", dpi=300, bbox_inches="tight")
plt.show()

# 4) Training loss curve
plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="Train Loss", linewidth=2)
plt.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
plt.title("Enhanced LSTM Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("models/LSTM/lstm_loss.png", dpi=300, bbox_inches="tight")
plt.show()

# ================= SAVE MODEL & SCALER ================= #
model.save("models/LSTM/lstm_nifty50_model.keras") 
joblib.dump(scaler, "models/LSTM/lstm_close_scaler.pkl")

print("\n✅ ENHANCED LSTM COMPLETE! All files saved in models/LSTM/:")
print("- lstm_nifty50_model.h5 (trained model)")
print("- lstm_close_scaler.pkl (scaler)")
print("- lstm_price_predictions_test.csv (test results)")
print("- lstm_future_30days_predictions.csv (future predictions)")
print("- lstm_prices.png (full test)")
print("- lstm_prices_last200.png (last 200 days)")
print("- lstm_future_forecast.png (with future)")
print("- lstm_loss.png (training curve)")

print(f"\n💡 Quick Load & Predict Next Day:")
print("```")
print("from tensorflow.keras.models import load_model")
print("import joblib")
print("model = load_model('models/LSTM/lstm_nifty50_model.h5')")
print("scaler = joblib.load('models/LSTM/lstm_close_scaler.pkl')")
print("```")
