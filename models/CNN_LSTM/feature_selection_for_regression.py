import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_regression

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

data = df[feature_cols + ["Close_next"]].dropna()
X = data[feature_cols]
y_reg = data["Close_next"]

# Pearson correlation with next close
corr_reg = X.corrwith(y_reg)

# Mutual information
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
mi_reg = mutual_info_regression(X_scaled, y_reg, random_state=42)

reg_table = (
    pd.DataFrame({
        "feature": feature_cols,
        "corr_with_next_close": corr_reg.values,
        "mutual_info": mi_reg
    })
    .sort_values("mutual_info", ascending=False)
)
print(reg_table)
#reg_table.to_csv("contents/reliance_feature_importance_regression.csv", index=False)
