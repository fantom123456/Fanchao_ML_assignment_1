"""
data_pipeline.py
Addresses Tasks (A) and (B): Preprocessing and Feature Engineering.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(filepath: str = "daily.csv"):
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 1. Calendar & Cyclical Transforms (Task B)
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["dayofyear"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Cyclical projection ensures smooth continuity (e.g., Dec 31 to Jan 1)
    df["sin_doy"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    df["sin_dow"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)

    # 2. Weather Interactions (Task B)
    df["temp_humidity_interaction"] = df["temp_c"] * (df["humidity_pct"] / 100.0)
    df["bad_weather_index"] = df["rain"] * df["wind_kmh"]

    # 3. Autoregressive Lag & Rolling Features (Task A & B)
    # Strictly shifted by at least 1 day so future labels do not leak
    df["lag_1"] = df["rentals"].shift(1)
    df["lag_2"] = df["rentals"].shift(2)
    df["lag_7"] = df["rentals"].shift(7)
    df["rolling_mean_7"] = df["rentals"].shift(1).rolling(7).mean()
    df["rolling_std_7"] = df["rentals"].shift(1).rolling(7).std()

    # Drop warmup rows from lagging
    df_clean = df.dropna().reset_index(drop=True)

    # 4. Temporal Split (2021-2022 for Training, 2023 for Testing)
    train_df = df_clean[df_clean["date"] < "2023-01-01"].copy()
    test_df = df_clean[df_clean["date"] >= "2023-01-01"].copy()

    feature_cols = [c for c in df_clean.columns if c not in ["date", "rentals", "dayofyear"]]

    X_train = train_df[feature_cols].copy()
    y_train = train_df["rentals"].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df["rentals"].copy()

    # 5. Fit scaler strictly on training distribution
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_cols, index=train_df.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols, index=test_df.index
    )

    return {
        "df_clean": df_clean,
        "train_df": train_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
    }


if __name__ == "__main__":
    data = load_and_preprocess_data()
    print("Preprocessed successfully.")
    print(f"Features ({len(data['feature_cols'])}): {data['feature_cols']}")
    print(f"Train samples: {len(data['X_train'])}, Test samples: {len(data['X_test'])}")