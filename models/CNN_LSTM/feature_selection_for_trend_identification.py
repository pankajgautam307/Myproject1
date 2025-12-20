import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif

df = pd.read_csv("contents/RELIANCE_features_15years.csv", parse_dates=["Date"])

feature_cols = [
    "Open","High","Low","Volume",
    "Close","SMA_9","SMA_20","SMA_50",
    "RSI_14","MACD","MACD_hist","BB_percentB",
    "Stoch_K","ATR_14","Williams_R",
    "ADX_14","CCI_20","ROC_10","Volume_SMA_5",
]
target_col = "Close"

# Build t and t+1 targets
df = df.sort_values("Date")
df["Close_next"] = df[target_col].shift(-1)
df["ret_next"] = (df["Close_next"] / df[target_col]) - 1.0

# binary label (optionally use small threshold like 0.001 to ignore tiny moves)
df["trend_label"] = (df["ret_next"] > 0).astype(int)

data_cls = df[feature_cols + ["trend_label"]].dropna()
X_cls = data_cls[feature_cols]
y_cls = data_cls["trend_label"]

# correlation of each feature with label (point-biserial = Pearson here)
corr_cls = X_cls.corrwith(y_cls)

# mutual information for classification
scaler = MinMaxScaler()
X_cls_scaled = scaler.fit_transform(X_cls)
mi_cls = mutual_info_classif(X_cls_scaled, y_cls, random_state=42)

cls_table = (
    pd.DataFrame({
        "feature": feature_cols,
        "corr_with_trend": corr_cls.values,
        "mutual_info": mi_cls
    })
    .sort_values("mutual_info", ascending=False)
)
print(cls_table)
cls_table.to_csv("reliance_feature_importance_classification.csv", index=False)
