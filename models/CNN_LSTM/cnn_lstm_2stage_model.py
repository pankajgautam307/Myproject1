import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    LSTM,
    Dense,
    Dropout,
    Concatenate,
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import joblib
import os


# =============== METRICS ================= #
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


print("🚀 Two‑Stage RELIANCE Model – Trend Classification + Price Regression")
print("=" * 100)

os.makedirs("models/CNN_LSTM", exist_ok=True)

# =============== LOAD DATA ================= #
df = pd.read_csv(
    "contents/RELIANCE_features_15years.csv",  # adjust filename as needed
    parse_dates=["Date"],
)

df = df.sort_values("Date").reset_index(drop=True)

# ---- Fill in your selected features here ---- #
classification_features = [
    # example; REPLACE with your chosen set
    "Close","High",
    "SMA_20",
    "SMA_50",
    "ROC_10",
    "Volume_SMA_5",
    "MACD_hist",
    "SMA_9",
    "MACD",
    "Low",
    "Open", "Stoch_K", "Williams_R",
    "RSI_14", "Volume", "CCI_20",
    "BB_percentB", "ATR_14", "ADX_14"
]

regression_features = [
    # example; REPLACE with your chosen set
    "Close",
    "SMA_20",
    "SMA_50",
    "ROC_10",
    "Volume_SMA_5",
    "MACD_hist",
    "SMA_9",
    "MACD",
    "Low",
    "Open",
    "RSI_14",
    "BB_percentB"
]

target_col = "Close"

# =============== BUILD TARGETS =============== #
# Next-day close
df["Close_next"] = df[target_col].shift(-1)

# Next-day return and trend label
df["ret_next"] = (df["Close_next"] / df[target_col]) - 1.0
# simple label: 1 if up, 0 otherwise; optionally add threshold
df["trend_label"] = (df["ret_next"] > 0).astype(int)

# Drop rows with NaNs in features or targets
all_needed_cols = (
    list(set(classification_features + regression_features))
    + [target_col, "Close_next", "ret_next", "trend_label"]
)
df = df[all_needed_cols + ["Date"]].dropna().reset_index(drop=True)

print("Final data shape:", df.shape)

# =============== TRAIN / TEST SPLIT =============== #
split_idx = int(0.8 * len(df))
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

dates = df["Date"]
# drop exact duplicate columns by name
df = df.loc[:, ~df.columns.duplicated()]
print("Columns after dropping duplicates:", df.columns.tolist())

print("Classification features:", classification_features)
print("Number of classification features:", len(classification_features))
print("df[classification_features].shape:", df[classification_features].shape)

# =============== STAGE 1: TREND CLASSIFIER =============== #
TIME_STEPS = 60

# ---- Scale classification features ---- #
cls_scaler = MinMaxScaler()
X_cls_all = cls_scaler.fit_transform(df[classification_features].values)

def create_cls_sequences(X_scaled, labels, time_steps):
    X_seq, y_seq = [], []
    for i in range(time_steps, len(X_scaled)):
        X_seq.append(X_scaled[i - time_steps : i])
        y_seq.append(labels[i])
    return np.array(X_seq), np.array(y_seq)

y_cls_all = df["trend_label"].values
X_cls, y_cls = create_cls_sequences(X_cls_all, y_cls_all, TIME_STEPS)

# Align train/test on sequences
split_seq = int(0.8 * len(X_cls))
X_cls_train, X_cls_test = X_cls[:split_seq], X_cls[split_seq:]
y_cls_train, y_cls_test = y_cls[:split_seq], y_cls[split_seq:]

print("Classifier data:", X_cls_train.shape, X_cls_test.shape)

# ---- Define classifier model (CNN–LSTM) ---- #
inputs_cls = Input(shape=(TIME_STEPS, len(classification_features)))

x = Conv1D(filters=32, kernel_size=3, padding="causal", activation="relu")(inputs_cls)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(filters=64, kernel_size=3, padding="causal", activation="relu")(x)

x = LSTM(64, return_sequences=False)(x)
x = Dropout(0.3)(x)
x = Dense(32, activation="relu")(x)
x = Dropout(0.3)(x)

out_cls = Dense(1, activation="sigmoid")(x)

cls_model = Model(inputs=inputs_cls, outputs=out_cls)
cls_model.compile(
    optimizer=Adam(learning_rate=1e-4, clipnorm=1.0),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

print("\n✅ Trend classifier compiled!")
cls_model.summary()

early_stopping_cls = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
)

history_cls = cls_model.fit(
    X_cls_train,
    y_cls_train,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stopping_cls],
    verbose=1,
)

print("✅ Trend classifier trained!")

# Evaluate classifier
y_prob_test = cls_model.predict(X_cls_test, verbose=0).flatten()
y_pred_test = (y_prob_test >= 0.5).astype(int)

acc_cls = accuracy_score(y_cls_test, y_pred_test)
cm_cls = confusion_matrix(y_cls_test, y_pred_test)

print("\n🏆 Trend classifier (test):")
print("--------------------------")
print(f"Accuracy: {acc_cls*100:.2f}%")
print("Confusion matrix:")
print(cm_cls)
print("\nClassification report:")
print(classification_report(y_cls_test, y_pred_test, digits=3))

# =============== STAGE 2: PRICE REGRESSOR (WITH TREND PROB) =============== #
# We use regression features + classifier's predicted prob as extra channel.

# ---- Scale regression features ---- #
reg_scaler = MinMaxScaler()
X_reg_all = reg_scaler.fit_transform(df[regression_features].values)

# ---- Scale target (Close_next) for stability ---- #
y_reg_all = df["Close_next"].values.reshape(-1, 1)
target_scaler = MinMaxScaler()
y_reg_scaled_all = target_scaler.fit_transform(y_reg_all)  # (N,1)

# ---- Build sequences with trend probability ---- #
# First compute classifier probabilities for *all* timesteps

# Need classifier input aligned with full data
X_cls_full_for_prob, _ = create_cls_sequences(X_cls_all, y_cls_all, TIME_STEPS)
# Get probabilities for all usable timesteps
p_up_all = cls_model.predict(X_cls_full_for_prob, verbose=0).flatten()  # length N - TIME_STEPS

def create_reg_sequences(X_reg_scaled, y_scaled, p_up, time_steps):
    """
    X_reg_scaled: (N, n_reg_features)
    y_scaled: (N, 1) scaled next Close
    p_up: (N - time_steps,) classifier probs aligned with i in [time_steps..N-1]

    Returns:
        X_seq: (N - time_steps, time_steps, n_reg_features + 1)
               where the extra feature is p_up broadcast over the whole window
        y_seq: (N - time_steps,)
    """
    X_seq, y_seq = [], []
    n_reg_features = X_reg_scaled.shape[1]
    for i in range(time_steps, len(X_reg_scaled)):
        window_feat = X_reg_scaled[i - time_steps : i]  # (T, n_reg_features)
        prob = p_up[i - time_steps]                     # scalar

        # extra feature channel = same prob for all timesteps in the window
        prob_col = np.full((time_steps, 1), prob, dtype=window_feat.dtype)

        # concatenate along feature dimension -> (T, n_reg_features + 1)
        window_feat_with_prob = np.concatenate([window_feat, prob_col], axis=1)

        X_seq.append(window_feat_with_prob)
        y_seq.append(y_scaled[i, 0])

    return np.array(X_seq), np.array(y_seq)


# Extend regression features with a dummy prob column for scaling of shapes
# (we'll overwrite prob at sequence-building time; scaler is only for reg features).
n_reg_features = len(regression_features)
# p_up vector length = len(df) - TIME_STEPS
X_reg_seq, y_reg_seq = create_reg_sequences(
    X_reg_all, y_reg_scaled_all, p_up_all, TIME_STEPS
)

print("\nRegressor data with trend prob:", X_reg_seq.shape, y_reg_seq.shape)

# Align train/test on sequences for regressor (same split index as classifier)
split_reg = split_seq  # use same fraction
X_reg_train, X_reg_test = X_reg_seq[:split_reg], X_reg_seq[split_reg:]
y_reg_train, y_reg_test = y_reg_seq[:split_reg], y_reg_seq[split_reg:]

# Dates for test sequences
dates_seq = dates[TIME_STEPS:]
dates_reg_test = dates_seq[split_reg:]

# ---- Define regression model (CNN–LSTM) ---- #
n_reg_input_features = X_reg_train.shape[2]  # = len(regression_features) + 1 (prob)

inputs_reg = Input(shape=(TIME_STEPS, n_reg_input_features))

xr = Conv1D(filters=32, kernel_size=3, padding="causal", activation="relu")(inputs_reg)
xr = MaxPooling1D(pool_size=2)(xr)
xr = Conv1D(filters=64, kernel_size=3, padding="causal", activation="relu")(xr)

xr = LSTM(64, return_sequences=False)(xr)
xr = Dropout(0.3)(xr)
xr = Dense(32, activation="relu")(xr)
xr = Dropout(0.3)(xr)

out_reg = Dense(1)(xr)  # scaled next-day Close

reg_model = Model(inputs=inputs_reg, outputs=out_reg)
reg_model.compile(
    optimizer=Adam(learning_rate=1e-4, clipnorm=1.0),
    loss="mse",
)

print("\n✅ Regression model (with trend prob) compiled!")
reg_model.summary()

early_stopping_reg = EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
)

history_reg = reg_model.fit(
    X_reg_train,
    y_reg_train,
    epochs=150,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stopping_reg],
    verbose=1,
)

print("✅ Regression model trained!")

# =============== EVALUATE REGRESSOR =============== #
y_pred_scaled = reg_model.predict(X_reg_test, verbose=0)
y_test_price = target_scaler.inverse_transform(y_reg_test.reshape(-1, 1))[:, 0]
y_pred_price = target_scaler.inverse_transform(y_pred_scaled)[:, 0]

metrics = stock_metrics_price(y_test_price, y_pred_price)

print("\n🏆 Two‑Stage RELIANCE – Price Regression with Trend Prob (Test):")
print("-" * 70)
for k, v in metrics.items():
    print(f"{k:<12}: {v}")

# =============== PLOTS =============== #
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Price prediction
axes[0, 0].plot(dates_reg_test, y_test_price, "g-", label="Actual Close", lw=2)
axes[0, 0].plot(dates_reg_test, y_pred_price, "r-", label="Pred Close", lw=2)
axes[0, 0].set_title("RELIANCE Close – Two‑Stage Model")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(dates_reg_test[-200:], y_test_price[-200:], "g-", lw=2)
axes[0, 1].plot(dates_reg_test[-200:], y_pred_price[-200:], "r-", lw=2)
axes[0, 1].set_title("Last 200 Days (Test)")
axes[0, 1].grid(True, alpha=0.3)

# Classifier training curves
axes[1, 0].plot(history_cls.history["loss"], label="Cls Train Loss", lw=2)
axes[1, 0].plot(history_cls.history["val_loss"], label="Cls Val Loss", lw=2)
axes[1, 0].set_title("Trend Classifier Loss")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Regressor training curves
axes[1, 1].plot(history_reg.history["loss"], label="Reg Train Loss", lw=2)
axes[1, 1].plot(history_reg.history["val_loss"], label="Reg Val Loss", lw=2)
axes[1, 1].set_title("Price Regressor Loss")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    "models/CNN_LSTM/reliance_two_stage_results.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# =============== SAVE MODELS & SCALERS =============== #
cls_model.save("models/CNN_LSTM/reliance_trend_classifier.keras")
reg_model.save("models/CNN_LSTM/reliance_price_regressor.keras")

joblib.dump(cls_scaler, "models/CNN_LSTM/classifier_feature_scaler.pkl")
joblib.dump(reg_scaler, "models/CNN_LSTM/regressor_feature_scaler.pkl")
joblib.dump(target_scaler, "models/CNN_LSTM/target_scaler.pkl")

results_df = pd.DataFrame(
    {
        "Date": dates_reg_test,
        "Actual_Close": y_test_price,
        "Pred_Close": y_pred_price,
        "Cls_Prob_Up": y_prob_test[-len(y_pred_price):],  # approx alignment
    }
)
results_df.to_csv(
    "models/CNN_LSTM/reliance_two_stage_predictions.csv",
    index=False,
)

print("\n✅ Two‑Stage RELIANCE pipeline COMPLETE! Files saved in models/CNN_LSTM")
