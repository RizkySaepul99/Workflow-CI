import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import json
import numpy as np
import os
  
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

run_id = os.environ.get("MLFLOW_RUN_ID")

# LOAD DATA
df = pd.read_csv("EAFC26_preprocessing.csv")

X = df.drop(columns=["OVR"])
y = df["OVR"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10],
    "min_samples_split": [2, 5]
}

rf = RandomForestRegressor(random_state=42)

grid = GridSearchCV(
    rf,
    param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1
)

if run_id:
    with mlflow.start_run(run_id=run_id):
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        
        y_pred = best_model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # LOG METRICS
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        # LOG PARAMS
        mlflow.log_params(grid.best_params_)
        
        # LOG MODEL
        mlflow.sklearn.log_model(best_model, "model")
        
        # ARTIFACT
        plt.figure(figsize=(8, 5))
        pd.Series(best_model.feature_importances_, index=X.columns)\
          .sort_values(ascending=False).head(10).plot(kind="bar")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        plt.close()
        
        mlflow.log_artifact("feature_importance.png")
        
        with open("model_metadata.json", "w") as f:
            json.dump(grid.best_params_, f, indent=2)
        
        mlflow.log_artifact("model_metadata.json")
else:
    mlflow.set_experiment("EAFC26_OVR_Advanced")
    
    with mlflow.start_run():
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        
        y_pred = best_model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # LOG METRICS
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        # LOG PARAMS
        mlflow.log_params(grid.best_params_)
        
        # LOG MODEL
        mlflow.sklearn.log_model(best_model, "model")
        
        # ARTIFACT
        plt.figure(figsize=(8, 5))
        pd.Series(best_model.feature_importances_, index=X.columns)\
          .sort_values(ascending=False).head(10).plot(kind="bar")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        plt.close()
        
        mlflow.log_artifact("feature_importance.png")
        
        with open("model_metadata.json", "w") as f:
            json.dump(grid.best_params_, f, indent=2)
        
        mlflow.log_artifact("model_metadata.json")
