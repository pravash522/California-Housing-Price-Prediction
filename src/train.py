import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from data_preprocessing import load_data, preprocess_data


def train_models():
    # Load dataset
    df = load_data()

    # Preprocess data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(df)

    # Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)
    linear_predictions = linear_model.predict(X_test_scaled)

    # Ridge Regression
    ridge_model = RidgeCV(
        alphas=np.logspace(-2, 2, 50),
        cv=5
    )
    ridge_model.fit(X_train_scaled, y_train)
    ridge_predictions = ridge_model.predict(X_test_scaled)

    # Lasso Regression
    lasso_model = LassoCV(
        cv=5,
        random_state=42
    )
    lasso_model.fit(X_train_scaled, y_train)
    lasso_predictions = lasso_model.predict(X_test_scaled)

    return {
        "linear_model": linear_model,
        "ridge_model": ridge_model,
        "lasso_model": lasso_model,
        "linear_predictions": linear_predictions,
        "ridge_predictions": ridge_predictions,
        "lasso_predictions": lasso_predictions,
        "y_test": y_test,
        "scaler": scaler
    }


if __name__ == "__main__":
    results = train_models()

    os.makedirs("models", exist_ok=True)
    joblib.dump(results["linear_model"], "models/linear_regression.pkl")
    joblib.dump(results["ridge_model"], "models/ridge_regression.pkl")
    joblib.dump(results["lasso_model"], "models/lasso_regression.pkl")

    print("Models trained successfully.")
    print("Best Ridge alpha:", results["ridge_model"].alpha_)
    print("Best Lasso alpha:", results["lasso_model"].alpha_)