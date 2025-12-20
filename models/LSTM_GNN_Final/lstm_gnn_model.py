import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Layer
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


print("🚀 BULLETPROOF LSTM-GNN HYBRID – Nifty50 + Correlation Graph")
print("=" * 60)

os.makedirs("models/LSTM_GNN_Final", exist_ok=True)

# ================= LOAD DATA ================= #
# Load real Nifty50 data
df = pd.read_csv("contents/Nifty50_features_15years.csv", index_col=0, parse_dates=True)

# SIMULATED 5-stock universe (Nifty50 + correlated sectors)
data = pd.DataFrame(index=df.index)
data["NIFTY50"] = df["Close"].copy()
data["BANKNIFTY"] = df["Close"] * 0.95 + np.random.normal(0, 50, len(df))
data["IT"] = df["Close"] * 0.92 + np.random.normal(0, 70, len(df))
data["PHARMA"] = df["Close"] * 0.88 + np.random.normal(0, 60, len(df))
data["AUTO"] = df["Close"] * 0.90 + np.random.normal(0, 55, len(df))

stocks = ["NIFTY50", "BANKNIFTY", "IT", "PHARMA", "AUTO"]
print(f"📈 Multi-stock universe: {stocks}")

# Train/test split
split = int(0.8 * len(data))
train_data = data.iloc[:split]
test_data = data.iloc[split:]


# ================= GRAPH CONSTRUCTION ================= #
def build_graph_adj(data, threshold=0.3):
    corr_matrix = data.corr().values
    adj_matrix = np.where(np.abs(corr_matrix) > threshold, np.abs(corr_matrix), 0.0)
    np.fill_diagonal(adj_matrix, 0.0)
    return adj_matrix.astype(np.float32)


graph_adj = build_graph_adj(train_data[stocks])
print(f"✅ Graph built: {graph_adj.shape} correlation matrix")


# ================= SCALING ================= #
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data[stocks])
test_scaled = scaler.transform(test_data[stocks])


# ================= SEQUENCE CREATION ================= #
TIME_STEPS = 30


def create_sequences_multi(series, time_steps):
    X, y = [], []
    for i in range(time_steps, len(series)):
        X.append(series[i - time_steps : i])
        # Predict NIFTY50 (index 0)
        y.append(series[i, 0])
    return np.array(X), np.array(y)


X_train, y_train = create_sequences_multi(train_scaled, TIME_STEPS)
X_test, y_test = create_sequences_multi(test_scaled, TIME_STEPS)

print(f"✅ Data ready: X_train {X_train.shape}")


# ================= FIXED GRAPH CONV LAYER ================= #
class GraphConvLayer(Layer):
    """
    Simple graph aggregation layer:
    - Input: node_features of shape (batch, n_nodes)
    - Uses fixed adjacency matrix A of shape (n_nodes, n_nodes)
    - Output: graph-level scalar feature of shape (batch, 1)
    """

    def __init__(self, adj_matrix, **kwargs):
        super(GraphConvLayer, self).__init__(**kwargs)
        self.adj_matrix = tf.constant(adj_matrix, dtype=tf.float32)

    def call(self, node_features):
        # node_features: (batch, n_nodes)  -> here n_nodes = len(stocks) = 5
        # Add feature dim: (batch, n_nodes, 1)
        x = tf.expand_dims(node_features, axis=-1)

        batch_size = tf.shape(node_features)[0]
        # Tile adjacency for batch: (batch, n_nodes, n_nodes)
        adj_tiled = tf.tile(self.adj_matrix[None, :, :], [batch_size, 1, 1])

        # Graph propagation: (batch, n_nodes, n_nodes) @ (batch, n_nodes, 1)
        x = tf.matmul(adj_tiled, x)  # (batch, n_nodes, 1)

        # Global average pooling over nodes and features -> (batch,)
        x = tf.reduce_mean(x, axis=[1, 2])

        # Return as (batch, 1) so it can be concatenated
        return tf.expand_dims(x, axis=-1)

    def get_config(self):
        config = super(GraphConvLayer, self).get_config()
        # adj_matrix is constant; not strictly needed to serialize for your use
        return config


# ================= LSTM-GNN HYBRID MODEL ================= #
inputs = Input(shape=(TIME_STEPS, len(stocks)))

# LSTM Branch: Temporal patterns across all 5 series
lstm_out = LSTM(50, return_sequences=False)(inputs)
lstm_out = Dropout(0.2)(lstm_out)

# Graph Branch: Inter-stock correlations using last timestep
last_timestep = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(inputs)  # (batch, 5)
graph_features = GraphConvLayer(graph_adj)(last_timestep)  # (batch, 1)

# Combine branches
combined = tf.keras.layers.Concatenate(axis=-1)([lstm_out, graph_features])
combined = Dense(25, activation="relu")(combined)
combined = Dropout(0.2)(combined)
output = Dense(1)(combined)

model = Model(inputs=inputs, outputs=output)
model.compile(optimizer="adam", loss="mse")

print("✅ LSTM-GNN Hybrid model compiled!")
model.summary()


# ================= TRAINING ================= #
early_stopping = EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
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
print("✅ LSTM-GNN Hybrid trained successfully!")


# ================= PREDICTION ================= #
y_pred_scaled = model.predict(X_test, verbose=0)

# Inverse transform NIFTY50 only (first feature)
dummy_features = np.zeros((len(y_pred_scaled), len(stocks)))
dummy_features[:, 0] = y_pred_scaled.flatten()
y_pred = scaler.inverse_transform(dummy_features)[:, 0]

# Actual Nifty50 prices
actual_nifty = test_data["NIFTY50"].iloc[TIME_STEPS:].values
test_index = test_data.index[TIME_STEPS:]


# ================= METRICS ================= #
metrics = stock_metrics_price(actual_nifty, y_pred)
print("\n🏆 LSTM-GNN HYBRID vs YOUR GRU-Bi 0.88%:")
print("-" * 40)
for k, v in metrics.items():
    print(f"{k:<12}: {v}")


# ================= VISUALIZATION ================= #
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Full prediction
axes[0, 0].plot(test_index, actual_nifty, "g-", label="Actual NIFTY50", lw=2)
axes[0, 0].plot(test_index, y_pred, "r-", label="LSTM-GNN Hybrid", lw=2)
axes[0, 0].set_title("LSTM-GNN: NIFTY50 Prediction")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Last 200 days
axes[0, 1].plot(test_index[-200:], actual_nifty[-200:], "g-", lw=2)
axes[0, 1].plot(test_index[-200:], y_pred[-200:], "r-", lw=2)
axes[0, 1].set_title("Last 200 Days")
axes[0, 1].grid(True, alpha=0.3)

# Training curves
axes[1, 0].plot(history.history["loss"], label="Train Loss", lw=2)
axes[1, 0].plot(history.history["val_loss"], label="Val Loss", lw=2)
axes[1, 0].set_title("Training Curves")
axes[1, 0].legend()
axes[1, 0].grid(True)

# Correlation heatmap
im = axes[1, 1].imshow(graph_adj, cmap="hot", interpolation="none")
axes[1, 1].set_title("Stock Correlation Graph")
plt.colorbar(im, ax=axes[1, 1])

plt.tight_layout()
plt.savefig(
    "models/LSTM_GNN_Hybrid/lstm_gnn_results.png", dpi=300, bbox_inches="tight"
)
plt.show()


# ================= SAVE EVERYTHING ================= #
results_df = pd.DataFrame(
    {"Date": test_index, "Actual_NIFTY50": actual_nifty, "LSTM_GNN_Pred": y_pred}
)
results_df.to_csv("models/LSTM_GNN_Final/lstm_gnn_predictions.csv", index=False)

model.save("models/LSTM_GNN_Final/lstm_gnn_model.keras")
joblib.dump(scaler, "models/LSTM_GNN_Final/lstm_gnn_scaler.pkl")
joblib.dump(graph_adj, "models/LSTM_GNN_Final/correlation_graph.pkl")

print("\n✅ LSTM-GNN HYBRID COMPLETE! Files saved:")
print("- lstm_gnn_predictions.csv")
print("- lstm_gnn_model.keras")
print("- lstm_gnn_results.png")
