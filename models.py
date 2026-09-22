"""
models.py
Addresses Tasks (C), (D), and (E): 5 Model Families, Objectives, Optimizers & Risk Tracking.
"""

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


def build_models():
    """Returns the dictionary of 5 model families with specified regularizers."""
    return {
        "1. Linear (Ridge L2)": {
            "model": Ridge(alpha=10.0),
            "use_scaled": True,
            "loss_description": "MSE + alpha * ||w||_2^2 (L2 penalty)",
            "optimizer": "Cholesky Decomposition / Conjugate Gradient",
        },
        "2. Tree Ensemble (HistGBR)": {
            "model": HistGradientBoostingRegressor(
                loss="squared_error",
                max_iter=150,
                learning_rate=0.05,
                random_state=42,
            ),
            "use_scaled": False,
            "loss_description": "Empirical MSE loss with L2 leaf shrinkage",
            "optimizer": "Gradient Tree Boosting (Newton-Raphson approximation)",
        },
        "3. Kernel Method (SVR RBF)": {
            "model": SVR(C=250.0, epsilon=15.0, kernel="rbf"),
            "use_scaled": True,
            "loss_description": "epsilon-Insensitive Loss + 0.5 * ||w||_2^2",
            "optimizer": "Sequential Minimal Optimization (SMO)",
        },
        "4. Neural Net (MLP)": {
            "model": MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=0.01,
                learning_rate_init=0.01,
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                n_iter_no_change=15,
            ),
            "use_scaled": True,
            "loss_description": "MSE Loss + L2 weight decay penalty",
            "optimizer": "Adam (Stochastic Gradient Descent with momentum)",
        },
        "5. Bayesian Learning (Bayes Ridge)": {
            "model": BayesianRidge(max_iter=300),
            "use_scaled": True,
            "loss_description": "Negative Marginal Log-Likelihood (Evidence Framework)",
            "optimizer": "L-BFGS / Evidence Maximization via iterative EM update",
        },
    }


def train_and_track_models(model_dict, data_dict):
    """Fits each model, tracking objective risk values and recording test predictions."""
    fitted_models = {}

    for name, config in model_dict.items():
        estimator = config["model"]
        X_tr = data_dict["X_train_scaled"] if config["use_scaled"] else data_dict["X_train"]
        X_te = data_dict["X_test_scaled"] if config["use_scaled"] else data_dict["X_test"]
        y_tr = data_dict["y_train"]

        estimator.fit(X_tr, y_tr)
        preds = estimator.predict(X_te)

        # Non-negativity clip (bike rentals cannot be negative)
        preds = [max(0.0, float(p)) for p in preds]

        # Extract risk and objective tracking values where available
        risk_info = {}
        if hasattr(estimator, "loss_curve_"):
            risk_info["final_train_loss"] = estimator.loss_curve_[-1]
            risk_info["loss_curve"] = estimator.loss_curve_
        if hasattr(estimator, "scores_"):
            risk_info["log_marginal_likelihood_curve"] = estimator.scores_
        if hasattr(estimator, "train_score_"):
            risk_info["gbr_loss_curve"] = estimator.train_score_

        fitted_models[name] = {
            "estimator": estimator,
            "predictions": preds,
            "risk_info": risk_info,
            "use_scaled": config["use_scaled"],
            "description": config["loss_description"],
            "optimizer": config["optimizer"],
        }

    return fitted_models