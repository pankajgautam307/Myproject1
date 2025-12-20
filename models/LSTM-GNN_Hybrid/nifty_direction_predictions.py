import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import os

print("🚀 NIFTY50 Direction Prediction – LSTM on 6 Indices (Log Returns)")
print("=" * 80)

os.makedirs("models/NIFTY_Direction_LSTM", exist_ok=True)

# ================= LOAD & ALIGN PRICE DATA ================= #
df_nifty = pd.read_csv("contents/Nifty50_features_15years.csv", index_col=0, parse_dates=True)
df_banknifty = pd.read_csv("contents/BankNifty_features_15years.csv", index_col=0, parse_dates=True)
df_niftyauto = pd.read_csv("contents/NIFTYAUTO_features_15years.csv", index_col=0, parse_dates=True)
df_niftyit = pd.read_csv("contents/NIFTYIT_features_15years.csv", index_col=0, parse_dates=True)
df_niftymetal = pd.read_csv("contents/NIFTYMETAL_features_15years.csv", index_col=0, parse_dates=True)
df_niftypharma = pd.read_csv("contents/NIFTYPHARMA_features_15years.csv", index_col=0, parse_dates=True)

prices = (
    df_nifty[["Close"]].rename(columns={"Close": "NIFTY50"})
    .join(df_banknifty[["Close"]].rename(columns={"Close": "BANKNIFTY"}), how="inner")
    .join(df_niftyit[["Close"]].rename(columns={"Close": "IT"}), how="inner")
    .join(df_niftypharma[["Close"]].rename(columns={"Close": "PHARMA"}), how="inner")
    .join(df_niftyauto[["Close"]].rename(columns={"Close": "AUTO"}), how="inner")
    .join(df_niftymetal[["Close"]].rename(columns={"Close": "METAL"}), how="inner")
)

stocks = ["NIFTY50", "BANKNIFTY", "IT", "PHARMA", "AUTO", "METAL"]
print(f"📈 Multi-index universe: {stocks}")
print("Aligned price data shape:", prices.shape)

# ================= LOG RETURNS ================= #
log_prices = np.log(prices)
log_returns = log_prices.diff().dropna()
prices = prices.loc[log_returns.index]

print("Sample NIFTY50 returns:", log_returns["NIFTY50"].head().to_list())

# ================= LABELS: NEXT-DAY DIRECTION ================= #
# sign of next-day NIFTY50 return: 1 if >0, 0 if <=0
nifty_ret = log_returns["NIFTY50"].values
y_dir = (nifty_ret > 0).astype(int)

# we will build sequences, so we trim later

# ================= TRAIN / TEST SPLIT ================= #
split = int(0.8 * len(log_returns))
train_ret = log_returns.iloc[:split]
test_ret = log_returns.iloc[split:]

train_y_raw = y_dir[:split]
test_y_raw = y_dir[split:]

TIME_STEPS = 60

def create_sequences_for_classification(df_ret, labels, time_steps):
    arr = df_ret.values  # shape (T, 6)
    X, y = [], []
    for i in range(time_steps, len(arr)):
        X.append(arr[i - time_steps : i])  # 60x6
        y.append(labels[i])               # direction at time i
    return np.array(X), np.array(y)

X_train, y_train = create_sequences_for_classification(
    train_ret[stocks], train_y_raw, TIME_STEPS
)
X_test, y_test = create_sequences_for_classification(
    test_ret[stocks], test_y_raw, TIME_STEPS
)

print(f"✅ Data ready: X_train {X_train.shape}, X_test {X_test.shape}")
print("Class balance (train):", np.bincount(y_train))

# ================= LSTM CLASSIFIER ================= #
inputs = Input(shape=(TIME_STEPS, len(stocks)))
x = LSTM(128, return_sequences=False)(inputs)
x = Dropout(0.3)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inputs, outputs=output)
optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

print("✅ NIFTY direction LSTM model compiled!")
model.summary()

# ================= TRAINING ================= #
early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
)

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1,
)

print("✅ NIFTY direction LSTM trained successfully!")

# ================= EVALUATION ================= #
y_prob = model.predict(X_test, verbose=0).flatten()
y_pred = (y_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n🏆 NIFTY50 Next-day Direction – Test Set")
print("----------------------------------------")
print(f"Accuracy : {acc*100:.2f}%")
print("Confusion matrix (rows: true, cols: pred):")
print(cm)
print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=3))

# ================= PLOTS ================= #
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Training curves
axes[0].plot(history.history["loss"], label="Train Loss", lw=2)
axes[0].plot(history.history["val_loss"], label="Val Loss", lw=2)
axes[0].set_title("Training Curves (Loss)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history["accuracy"], label="Train Acc", lw=2)
axes[1].plot(history.history["val_accuracy"], label="Val Acc", lw=2)
axes[1].set_title("Training Curves (Accuracy)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    "models/NIFTY_Direction_LSTM/nifty_direction_training.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# ================= SAVE MODEL & RESULTS ================= #
results_df = pd.DataFrame(
    {"y_true": y_test, "y_prob": y_prob, "y_pred": y_pred}
)
results_df.to_csv(
    "models/NIFTY_Direction_LSTM/nifty_direction_predictions.csv", index=False
)

model.save("models/NIFTY_Direction_LSTM/nifty_direction_lstm.keras")

print("\n✅ NIFTY Direction LSTM COMPLETE! Files saved:")
print("- nifty_direction_lstm.keras")
print("- nifty_direction_predictions.csv")
print("- nifty_direction_training.png")
