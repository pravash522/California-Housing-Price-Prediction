import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from train import train_models


def evaluate_model(y_test, predictions):
    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mae, rmse, r2


def evaluate_all_models(results):

    y_test = results["y_test"]

    # Evaluate Linear Regression
    mae_linear, rmse_linear, r2_linear = evaluate_model(
        y_test,
        results["linear_predictions"]
    )

    # Evaluate Ridge Regression
    mae_ridge, rmse_ridge, r2_ridge = evaluate_model(
        y_test,
        results["ridge_predictions"]
    )

    # Evaluate Lasso Regression
    mae_lasso, rmse_lasso, r2_lasso = evaluate_model(
        y_test,
        results["lasso_predictions"]
    )

    # Create comparison table
    comparison = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Ridge Regression",
            "Lasso Regression"
        ],
        "MAE": [
            mae_linear,
            mae_ridge,
            mae_lasso
        ],
        "RMSE": [
            rmse_linear,
            rmse_ridge,
            rmse_lasso
        ],
        "R2 Score": [
            r2_linear,
            r2_ridge,
            r2_lasso
        ]
    })
    return comparison


def find_best_model(comparison):
    best_model = comparison.loc[
        comparison["R2 Score"].idxmax()
    ]
    return best_model


if __name__ == "__main__":

    results = train_models()
    comparison = evaluate_all_models(results)
    print("\nModel Comparison:")
    print(comparison)

    best_model = find_best_model(comparison)
    print("\nBest Performing Model:")
    print(best_model)

    os.makedirs("results", exist_ok=True)
    comparison.to_csv("results/model_comparison.csv",index=False)
    prediction_map = {
        "Linear Regression": results["linear_predictions"],
        "Ridge Regression": results["ridge_predictions"],
        "Lasso Regression": results["lasso_predictions"]
    }
    best_model.to_frame().T.to_csv("results/best-model.csv",index=False)