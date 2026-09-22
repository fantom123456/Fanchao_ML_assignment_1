"""
main.py
Root execution script for CPS485 Assignment #1.
Runs end-to-end pipeline, evaluates 5 model families, rolls out multi-step forecast,
and exports all diagnostic charts.
"""

from data_pipeline import load_and_preprocess_data
from evaluate import (
    evaluate_test_performance,
    generate_diagnostic_plots,
    multi_step_rollout,
)
from models import build_models, train_and_track_models


def main():
    print("=" * 80)
    print("CPS485: BIKE RENTAL FORECASTING EXPERIMENT PIPELINE")
    print("=" * 80)

    # 1. Pipeline & Features (Tasks A & B)
    print("\n[Stage 1] Loading data, engineering cyclical & lag features...")
    data = load_and_preprocess_data("daily.csv")
    print(f"Features: {len(data['feature_cols'])} | Train: {len(data['X_train'])} | Test: {len(data['X_test'])}")

    # 2. Build & Train Models (Tasks C, D, E)
    print("\n[Stage 2] Training 5 model families and tracking loss objectives...")
    model_definitions = build_models()
    fitted_models = train_and_track_models(model_definitions, data)

    # 3. Model Evaluation Table (Task F)
    print("\n[Stage 3] Model Evaluation Summary (Held-out 2023 Out-of-Sample Test Set):")
    perf_table = evaluate_test_performance(fitted_models, data["y_test"].values)
    print(perf_table[["Model Family", "RMSE", "MAE", "R2", "MAPE (%)"]].to_string(index=False))

    # 4. Multi-Step Horizon Forecast [t, ..., t+7]
    print("\n[Stage 4] Simulating 7-Day Recursive Multi-Step Horizon Forecast [t, ..., t+7]:")
    # Aligned exact dictionary key with models.py
    bayes_model = fitted_models["5. Bayesian Learning (Bayes Ridge)"]["estimator"]
    rollout_df = multi_step_rollout(
        bayes_model,
        data["scaler"],
        data["train_df"],
        data["test_df"],
        data["feature_cols"],
        horizon=7,
    )
    print(rollout_df.to_string(index=False))

    # 5. Diagnostic Figures (Tasks F, G & I)
    print("\n[Stage 5] Generating diagnostic plots for paper defense...")
    generate_diagnostic_plots(fitted_models, data, output_dir="figures")

    print("\n" + "=" * 80)
    print("Execution complete. All deliverables ready for GitHub commit.")
    print("=" * 80)


if __name__ == "__main__":
    main()