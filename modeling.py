from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

import mlflow


def compare_models(X_train, y_train, cv=10):
    mlflow.set_experiment("walmart_model_comparison__holiday_flag")

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'KNN': KNeighborsRegressor(),
        'Random Forest': RandomForestRegressor(random_state=42)
    }

    results = {}
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            r2_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
            mse_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
            mae_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error')

            mean_r2 = r2_scores.mean()
            mean_mse = (-mse_scores).mean()
            mean_mae = (-mae_scores).mean()

            # Log parameters
            mlflow.log_param("model", name)
            mlflow.log_param("cv_folds", cv)

            # Log metrics
            mlflow.log_metric("mean_r2", mean_r2)
            mlflow.log_metric("std_r2", r2_scores.std())
            mlflow.log_metric("mean_mse", mean_mse)
            mlflow.log_metric("mean_mae", mean_mae)

            results[name] = {
                'mean_r2': mean_r2,
                'std_r2': r2_scores.std(),
                'mean_mse': mean_mse,
                'mean_mae': mean_mae
            }
            print(
                f"{name:25s} → R²: {mean_r2:.4f} ± {r2_scores.std():.4f} | MSE: {mean_mse:,.0f} | MAE: {mean_mae:,.0f}")

    return results


def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    print("Model trained successfully!")
    return model

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"R²   : {r2:.4f}")
    print(f"MSE  : {mse:,.0f}")
    print(f"RMSE : {rmse:,.0f}")
    print(f"MAE  : {mae:,.0f}")

    return y_pred, r2, mse, mae

if __name__ == "__main__":
    import pandas as pd
    import dagshub
    import mlflow
    from dotenv import load_dotenv
    import os
    from data_engineering import todate, encode_store
    from feature_selection import compute_mutual_info, select_features_v2, add_holiday_interactions
    from processing import split_data, scale_features
    from sklearn.metrics import r2_score

    # ── Load credentials ──────────────────────────────────────────────────────
    load_dotenv()
    dagshub.auth.add_app_token(os.getenv("DagsHub_token"))
    dagshub.init("Walmart-sales-prediction", "emnaaS", mlflow=True)

    # ── Load & prepare data ───────────────────────────────────────────────────
    df = pd.read_csv('data/Walmart_Sales.csv')
    df = todate(df)
    df = encode_store(df)
    df = add_holiday_interactions(df)

    # ── Feature selection ─────────────────────────────────────────────────────
    mi_scores = compute_mutual_info(df)
    X, y = select_features_v2(df, mi_scores)

    # ── Split & scale ─────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # ── Compare models ────────────────────────────────────────────────────────
    compare_models(X_train_scaled, y_train)

    # ── Train & log final model ───────────────────────────────────────────────
    mlflow.set_experiment("walmart_final_model")
    with mlflow.start_run(run_name="RandomForestRegressor"):
        model = train_model(X_train_scaled, y_train)
        y_pred, r2, mse, mae = evaluate_model(model, X_test_scaled, y_test)

        mlflow.log_param("n_estimators", 150)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("r2",   r2)
        mlflow.log_metric("mse",  mse)
        mlflow.log_metric("mae",  mae)
        mlflow.sklearn.log_model(model, "random_forest_model")

    print("Pipeline complete!")