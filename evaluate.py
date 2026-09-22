"""
evaluate.py
Addresses Tasks (F), (G), (H), and (I): Rigorous Evaluation, Multi-step Horizons, and Diagnostics.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_test_performance(fitted_models, y_true):
    records = []
    for name, obj in fitted_models.items():
        preds = np.array(obj["predictions"])
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        mae = mean_absolute_error(y_true, preds)
        r2 = r2_score(y_true, preds)
        mape = np.mean(np.abs((y_true - preds) / np.maximum(y_true, 1))) * 100

        records.append({
            "Model Family": name,
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2),
            "R2": round(r2, 4),
            "MAPE (%)": round(mape, 2),
            "Objective Function": obj["description"],
            "Optimizer": obj["optimizer"],
        })
    return pd.DataFrame(records)


def multi_step_rollout(model, scaler, train_df, test_df, feature_cols, horizon: int = 7):
    """
    Forecasting horizon [t, ..., t+h]:
    Recursively rolls forward day-by-day without leaking future target values.
    """
    history = list(train_df["rentals"].values)
    horizon_preds = []

    for i in range(horizon):
        row = test_df.iloc[i].copy()

        # Autoregressive recursive feature updates
        row["lag_1"] = history[-1]
        row["lag_2"] = history[-2]
        row["lag_7"] = history[-7]
        row["rolling_mean_7"] = np.mean(history[-7:])
        row["rolling_std_7"] = np.std(history[-7:], ddof=1) if len(history[-7:]) > 1 else 0.0

        features_df = pd.DataFrame([row[feature_cols]], columns=feature_cols)
        features_scaled = pd.DataFrame(scaler.transform(features_df), columns=feature_cols)
        pred = max(0.0, float(model.predict(features_scaled)[0]))

        horizon_preds.append(pred)
        history.append(pred)

    actuals = test_df["rentals"].values[:horizon]
    return pd.DataFrame({
        "Date": test_df["date"].dt.strftime("%Y-%m-%d").values[:horizon],
        "Horizon": [f"t+{i+1}" for i in range(horizon)],
        "Predicted_Rentals": np.round(horizon_preds, 1),
        "Actual_Rentals": actuals,
        "Abs_Error": np.round(np.abs(horizon_preds - actuals), 1),
    })


def generate_diagnostic_plots(fitted_models, data_dict, output_dir: str = "figures"):
    os.makedirs(output_dir, exist_ok=True)
    y_test = data_dict["y_test"].values
    dates = data_dict["test_df"]["date"].values

    bayes_obj = fitted_models["5. Bayesian Learning (Bayes Ridge)"]
    bayes_model = bayes_obj["estimator"]
    X_test_scaled = data_dict["X_test_scaled"]

    # Posterior predictive uncertainty
    preds_mean, preds_std = bayes_model.predict(X_test_scaled, return_std=True)
    preds_mean = np.clip(preds_mean, 0, None)

    # 1. Timeline Forecast Plot
    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_test, label="Actual Rentals", color="black", alpha=0.6, linewidth=1.2)
    plt.plot(dates, preds_mean, label="Bayesian Ridge Forecast", color="#1f77b4", linewidth=1.5)
    plt.fill_between(
        dates,
        np.clip(preds_mean - 1.96 * preds_std, 0, None),
        preds_mean + 1.96 * preds_std,
        color="#1f77b4",
        alpha=0.25,
        label="95% Predictive Credible Interval",
    )
    plt.title("2023 Out-of-Sample Rentals: Actual vs. Bayesian Forecast")
    plt.xlabel("Date")
    plt.ylabel("Rentals")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig1_forecast_timeline.png", dpi=300)
    plt.close()

    # 2. Residual Distribution Plot (Error Checking)
    residuals = y_test - preds_mean
    plt.figure(figsize=(8, 4))
    plt.scatter(preds_mean, residuals, alpha=0.5, color="#2ca02c")
    plt.axhline(0, color="red", linestyle="--", linewidth=1.2)
    plt.title("Bayesian Ridge Residuals vs Fitted Values")
    plt.xlabel("Predicted Rentals")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig2_residual_analysis.png", dpi=300)
    plt.close()

    # 3. Permutation Feature Importance (Interpretation Task G)
    gbr_model = fitted_models["2. Tree Ensemble (HistGBR)"]["estimator"]
    perm = permutation_importance(
        gbr_model, data_dict["X_test"], data_dict["y_test"], n_repeats=10, random_state=42
    )
    sorted_idx = perm.importances_mean.argsort()[-10:]

    plt.figure(figsize=(9, 5))
    plt.barh(np.array(data_dict["feature_cols"])[sorted_idx], perm.importances_mean[sorted_idx], color="#ff7f0e")
    plt.title("Top 10 Feature Importances (Tree Ensemble Permutation)")
    plt.xlabel("Mean Drop in Test Accuracy")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig3_feature_importance.png", dpi=300)
    plt.close()

    # 4. Neural Network Training Loss Curve (Task E)
    mlp_model = fitted_models["4. Neural Net (MLP)"]["estimator"]
    if hasattr(mlp_model, "loss_curve_"):
        plt.figure(figsize=(7, 4))
        plt.plot(mlp_model.loss_curve_, color="#9467bd", linewidth=1.5)
        plt.title("MLP Training Risk / Loss vs. Iterations")
        plt.xlabel("Epochs")
        plt.ylabel("Training Loss")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/fig4_mlp_loss_curve.png", dpi=300)
        plt.close()

    print(f"Generated 4 diagnostic figures in '{output_dir}/'")