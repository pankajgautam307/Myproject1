import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Layer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
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


print("🚀 LSTM-GNN HYBRID – 6 Indices, RAW LOG RETURNS (no scaler)")
print("=" * 80)

os.makedirs("models/LSTM-GNN_Hybrid", exist_ok=True)


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


# ================= LOG RETURNS (UNSCALED) ================= #
log_prices = np.log(prices)
log_returns = log_prices.diff().dropna()
prices = prices.loc[log_returns.index]

print("Sample NIFTY50 returns:",
      log_returns["NIFTY50"].head().to_list())


# ================= TRAIN / TEST SPLIT ================= #
split = int(0.8 * len(log_returns))
train_ret = log_returns.iloc[:split]
test_ret = log_returns.iloc[split:]

train_price = prices.iloc[:split]
test_price = prices.iloc[split:]


# ================= GRAPH CONSTRUCTION (FROM TRAIN PRICES) ================= #
def build_graph_adj(data_prices, threshold=0.1):
    corr_matrix = data_prices.corr().values
    adj_matrix = np.where(np.abs(corr_matrix) > threshold, np.abs(corr_matrix), 0.0)
    np.fill_diagonal(adj_matrix, 0.0)
    return adj_matrix.astype(np.float32)


graph_adj = build_graph_adj(train_price[stocks])
print(f"✅ Graph built: {graph_adj.shape} correlation matrix (threshold=0.1)")


# ================= SEQUENCE CREATION ON UNSCALED RETURNS ================= #
TIME_STEPS = 60


def create_sequences_multi(df_ret, time_steps):
    arr = df_ret.values  # shape (T, 6)
    X, y = [], []
    for i in range(time_steps, len(arr)):
        X.append(arr[i - time_steps : i])
        y.append(arr[i, 0])  # NIFTY50 next-step return
    return np.array(X), np.array(y)


X_train, y_train = create_sequences_multi(train_ret[stocks], TIME_STEPS)
X_test, y_test = create_sequences_multi(test_ret[stocks], TIME_STEPS)

print(f"✅ Data ready (raw returns): X_train {X_train.shape}, y_train {y_train.shape}")
print("y_train stats:", y_train.mean(), y_train.std())


# ================= RICH GRAPH CONV LAYER ================= #
class GraphConvLayer(Layer):
    """
    Graph layer with per-node Dense -> A@X -> pooling -> Dense.
    Input: node_features (batch, n_nodes)
    Output: graph feature (batch, graph_units)
    """

    def __init__(self, adj_matrix, node_units=16, graph_units=32, **kwargs):
        super(GraphConvLayer, self).__init__(**kwargs)
        self.adj_matrix = tf.constant(adj_matrix, dtype=tf.float32)
        self.node_units = node_units
        self.graph_units = graph_units
        self.node_dense = Dense(node_units, activation="relu")
        self.graph_dense = Dense(graph_units, activation="relu")

    def call(self, node_features):
        batch_size = tf.shape(node_features)[0]
        x = tf.expand_dims(node_features, axis=-1)  # (batch, n_nodes, 1)
        x = self.node_dense(x)                     # (batch, n_nodes, node_units)
        adj_tiled = tf.tile(self.adj_matrix[None, :, :], [batch_size, 1, 1])
        x = tf.matmul(adj_tiled, x)                # (batch, n_nodes, node_units)
        x = tf.reduce_mean(x, axis=1)              # (batch, node_units)
        x = self.graph_dense(x)                    # (batch, graph_units)
        return x

    def get_config(self):
        config = super(GraphConvLayer, self).get_config()
        config.update(
            {"node_units": self.node_units, "graph_units": self.graph_units}
        )
        return config


# ================= LSTM-GNN HYBRID (RAW RETURNS) ================= #
inputs = Input(shape=(TIME_STEPS, len(stocks)))

lstm_out = LSTM(128, return_sequences=False)(inputs)
lstm_out = Dropout(0.3)(lstm_out)

last_timestep = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(inputs)
graph_features = GraphConvLayer(graph_adj, node_units=16, graph_units=32)(last_timestep)

combined = tf.keras.layers.Concatenate(axis=-1)([lstm_out, graph_features])
combined = Dense(64, activation="relu")(combined)
combined = Dropout(0.3)(combined)
output = Dense(1)(combined)   # directly predict log-return

model = Model(inputs=inputs, outputs=output)
optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)
model.compile(optimizer=optimizer, loss="mse")

print("✅ LSTM-GNN Hybrid (raw returns) model compiled!")
model.summary()


# ================= TRAINING ================= #
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

print("✅ LSTM-GNN Hybrid (raw returns) trained successfully!")


# ================= PREDICTION & PRICE RECONSTRUCTION ================= #
y_pred_ret = model.predict(X_test, verbose=0).flatten()

true_ret_unscaled = test_ret["NIFTY50"].iloc[TIME_STEPS:].values

def reconstruct_prices(start_price, returns):
    prices = [start_price]
    for r in returns:
        prices.append(prices[-1] * np.exp(r))
    return np.array(prices[1:])

start_price = test_price["NIFTY50"].iloc[TIME_STEPS - 1]
y_pred_price = reconstruct_prices(start_price, y_pred_ret)
actual_price = reconstruct_prices(start_price, true_ret_unscaled)

test_index = test_ret.index[TIME_STEPS:]

# ================= METRICS ================= #
metrics = stock_metrics_price(actual_price, y_pred_price)
print("\n🏆 LSTM-GNN HYBRID (raw returns) – Price reconstruction:")
print("-" * 55)
for k, v in metrics.items():
    print(f"{k:<12}: {v}")

# ================= VISUALIZATION ================= #
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].plot(test_index, actual_price, "g-", label="Actual NIFTY50", lw=2)
axes[0, 0].plot(test_index, y_pred_price, "r-", label="LSTM-GNN Hybrid", lw=2)
axes[0, 0].set_title("NIFTY50 Price from Predicted Returns")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(test_index[-200:], actual_price[-200:], "g-", lw=2)
axes[0, 1].plot(test_index[-200:], y_pred_price[-200:], "r-", lw=2)
axes[0, 1].set_title("Last 200 Days")
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(history.history["loss"], label="Train Loss", lw=2)
axes[1, 0].plot(history.history["val_loss"], label="Val Loss", lw=2)
axes[1, 0].set_title("Training Curves (Raw Returns)")
axes[1, 0].legend()
axes[1, 0].grid(True)

im = axes[1, 1].imshow(graph_adj, cmap="hot", interpolation="none")
axes[1, 1].set_title("Stock Correlation Graph (Prices)")
plt.colorbar(im, ax=axes[1, 1])

plt.tight_layout()
plt.savefig(
    "models/LSTM-GNN_Hybrid/lstm_gnn_results_rawreturns_6idx.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# ================= SAVE ================= #
results_df = pd.DataFrame(
    {
        "Date": test_index,
        "Actual_NIFTY50": actual_price,
        "LSTM_GNN_Pred": y_pred_price,
        "True_LogRet": true_ret_unscaled,
        "Pred_LogRet": y_pred_ret,
    }
)
results_df.to_csv(
    "models/LSTM-GNN_Hybrid/lstm_gnn_predictions_rawreturns_6idx.csv",
    index=False,
)

model.save("models/LSTM-GNN_Hybrid/lstm_gnn_model_rawreturns_6idx.keras")
joblib.dump(graph_adj, "models/LSTM-GNN_Hybrid/correlation_graph.pkl")

print("\n✅ LSTM-GNN HYBRID RAW RETURNS (6 indices) COMPLETE! Files saved:")
print("- lstm_gnn_predictions_rawreturns_6idx.csv")
print("- lstm_gnn_model_rawreturns_6idx.keras")
print("- lstm_gnn_results_rawreturns_6idx.png")
