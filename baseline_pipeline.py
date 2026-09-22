import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# 1. Load and parse dates
df = pd.read_csv("daily.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# 2. Calendar and cyclical features
df["dayofweek"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
df["sin_doy"] = np.sin(2 * np.pi * df["date"].dt.dayofyear / 365.25)
df["cos_doy"] = np.cos(2 * np.pi * df["date"].dt.dayofyear / 365.25)

# 3. Lag features (strictly past info: t-1, t-7)
df["lag_1"] = df["rentals"].shift(1)
df["lag_7"] = df["rentals"].shift(7)
df["rolling_mean_7"] = df["rentals"].shift(1).rolling(window=7).mean()

# Drop rows with NaN caused by shifting
df = df.dropna().reset_index(drop=True)

# 4. Chronological train/test split
train = df[df["date"] < "2023-01-01"]
test = df[df["date"] >= "2023-01-01"]

feature_cols = [c for c in df.columns if c not in ["date", "rentals"]]
X_train, y_train = train[feature_cols], train["rentals"]
X_test, y_test = test[feature_cols], test["rentals"]

# 5. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Fit 5 model families
models = {
    "Ridge (Linear)": (Ridge(alpha=10.0), X_train_scaled, X_test_scaled),
    "HistGBR (Ensemble)": (HistGradientBoostingRegressor(random_state=42), X_train, X_test),
    "SVR (Kernel)": (SVR(C=100.0, epsilon=10.0), X_train_scaled, X_test_scaled),
    "MLP (Neural Net)": (MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42), X_train_scaled, X_test_scaled),
    "Bayesian Ridge": (BayesianRidge(), X_train_scaled, X_test_scaled)
}

print(f"{'Model':<22} | {'RMSE':<8} | {'MAE':<8} | {'R2':<8}")
print("-" * 52)
for name, (model, X_tr, X_te) in models.items():
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{name:<22} | {rmse:<8.2f} | {mae:<8.2f} | {r2:<8.3f}")