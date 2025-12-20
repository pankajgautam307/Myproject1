import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import joblib
import os


# ================ METRICS ================= #
def stock_metrics_price(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
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


print("🚀 Multifeature CNN–LSTM – RELIANCE (Indicators → Next-day Close)")
print("=" * 90)

os.makedirs("models/CNN_LSTM", exist_ok=True)

# ================ LOAD DATA ================= #
df = pd.read_csv(
    "contents/RELIANCE_features_15years.csv",
    index_col=0,
    parse_dates=True,
)

# Ensure the expected columns exist
all_cols = [
    "Open", "High", "Low", "Close", "Volume",
    "Close_lag1", "Volume_lag1",
    "RSI_14", "SMA_20", "SMA_50",
    "MACD", "MACD_hist", "BB_percentB",
    "Stoch_K", "ATR_14", "Williams_R",
    "ADX_14", "CCI_20", "ROC_10", "Volume_SMA_20",
]
for c in all_cols:
    if c not in df.columns:
        raise ValueError(f"Missing column in CSV: {c}")

# Features for model input (exclude Close target)
feature_cols = [
    "Open", "High", "Low", "Volume",
    "Close_lag1", "Volume_lag1",
    "RSI_14", "SMA_20", "SMA_50",
    "MACD", "MACD_hist", "BB_percentB",
    "Stoch_K", "ATR_14", "Williams_R",
    "ADX_14", "CCI_20", "ROC_10", "Volume_SMA_20",
]
target_col = "Close"

data = df.copy().dropna()  # drop any initial NaNs from indicators
features = data[feature_cols].values
target = data[target_col].values
dates = data.index

print("Data shape (rows, features):", features.shape)

# ================ SCALE FEATURES & TARGET ================= #
feat_scaler = MinMaxScaler()
features_scaled = feat_scaler.fit_transform(features)

# Scale target separately (to keep magnitude controlled)
target_scaler = MinMaxScaler()
target_scaled = target_scaler.fit_transform(target.reshape(-1, 1))  # (T,1)

TIME_STEPS = 60  # 60-day window


def create_sequences_multifeature(X_scaled, y_scaled, time_steps):
    X_seq, y_seq = [], []
    for i in range(time_steps, len(X_scaled)):
        X_seq.append(X_scaled[i - time_steps : i])  # (time_steps, n_features)
        y_seq.append(y_scaled[i, 0])                # next-day scaled Close
    return np.array(X_seq), np.array(y_seq)


X, y = create_sequences_multifeature(features_scaled, target_scaled, TIME_STEPS)
print(f"✅ Sequences ready: X {X.shape}, y {y.shape}")

# Train/test split on sequences
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

test_dates = dates[TIME_STEPS:][split:]  # align dates with X_test

# ================ CNN–LSTM MODEL ================= #
n_features = X.shape[2]

inputs = Input(shape=(TIME_STEPS, n_features))

# CNN block: extract local temporal patterns per feature
x = Conv1D(filters=64, kernel_size=3, padding="causal", activation="relu")(inputs)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(filters=128, kernel_size=3, padding="causal", activation="relu")(x)

# LSTM block: long-term dependencies across all features
x = LSTM(128, return_sequences=False)(x)
x = Dropout(0.3)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)

output = Dense(1)(x)  # scaled next-day Close

model = Model(inputs=inputs, outputs=output)
optimizer = Adam(learning_rate=5e-4, clipnorm=1.0)
model.compile(optimizer=optimizer, loss="mse")

print("✅ Multifeature CNN–LSTM model compiled!")
model.summary()

# ================ TRAINING ================= #
early_stopping = EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
)

history = model.fit(
    X_train,
    y_train,
    epochs=150,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1,
)

print("✅ Multifeature CNN–LSTM model trained successfully!")

# ================ PREDICTION & INVERSE SCALE ================= #
y_pred_scaled = model.predict(X_test, verbose=0)

y_test_price = target_scaler.inverse_transform(y_test.reshape(-1, 1))[:, 0]
y_pred_price = target_scaler.inverse_transform(y_pred_scaled)[:, 0]

# ================ METRICS ================= #
metrics = stock_metrics_price(y_test_price, y_pred_price)
print("\n🏆 CNN–LSTM (multi-feature RELIANCE) – Test Set:")
print("-" * 60)
for k, v in metrics.items():
    print(f"{k:<12}: {v}")

# ================ VISUALIZATION ================= #
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].plot(test_dates, y_test_price, "g-", label="Actual Close", lw=2)
axes[0, 0].plot(test_dates, y_pred_price, "r-", label="CNN–LSTM Pred", lw=2)
axes[0, 0].set_title("RELIANCE Close – Multifeature CNN–LSTM")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(test_dates[-200:], y_test_price[-200:], "g-", lw=2)
axes[0, 1].plot(test_dates[-200:], y_pred_price[-200:], "r-", lw=2)
axes[0, 1].set_title("Last 200 Days")
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(history.history["loss"], label="Train Loss", lw=2)
axes[1, 0].plot(history.history["val_loss"], label="Val Loss", lw=2)
axes[1, 0].set_title("Training Curves (MSE)")
axes[1, 0].legend()
axes[1, 0].grid(True)

residuals = y_test_price - y_pred_price
axes[1, 1].hist(residuals, bins=50, color="steelblue", alpha=0.7)
axes[1, 1].set_title("Prediction Residuals")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    "models/CNN_LSTM/cnn_lstm_results.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# ================ SAVE EVERYTHING ================= #
results_df = pd.DataFrame(
    {"Date": test_dates, "Actual_Close": y_test_price, "Pred_Close": y_pred_price}
)
results_df.to_csv(
    "models/CNN_LSTM/cnn_lstm__predictions.csv",
    index=False,
)

model.save("models/CNN_LSTM/cnn_lstm_model.keras")
joblib.dump(feat_scaler, "models/CNN_LSTM/feature_scaler.pkl")
joblib.dump(target_scaler, "models/CNN_LSTM/target_scaler.pkl")

print("\n✅ CNN–LSTM -FEATURE COMPLETE! Files saved:")
print("- cnn_lstmpredictions.csv")
print("- cnn_lstm_model.keras")
print("- cnn_lstm_results.png")
