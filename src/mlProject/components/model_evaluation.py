import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.utils.common import save_json
from pathlib import Path

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config


    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2



    def log_into_mlflow(self):


        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]


        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


        # # Check environment variables and configuration
        # print("Environment variables and MLflow configuration:")
        # print("MLFLOW_TRACKING_URI:", os.getenv('MLFLOW_TRACKING_URI'))
        # print("MLFLOW_TRACKING_USERNAME:", os.getenv('MLFLOW_TRACKING_USERNAME'))
        # print("MLFLOW_TRACKING_PASSWORD:", os.getenv('MLFLOW_TRACKING_PASSWORD'))
        # print("MLflow tracking URI:", mlflow.get_tracking_uri())
        # print("Tracking URL type store:", tracking_url_type_store)
        # print("Model path:", self.config.model_path)
        # print("Test data path:", self.config.test_data_path)
        # print("Metric file name:", self.config.metric_file_name)
        # print("Target column:", self.config.target_column)
        # print("MLflow URI:", self.config.mlflow_uri)
        # print("All parameters:", self.config.all_params)

        with mlflow.start_run():
            print("MLflow run started.")
            predicted_qualities = model.predict(test_x)
            print("Model prediction completed.")

            (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)
            print(f"Evaluation metrics calculated: RMSE={rmse}, MAE={mae}, R2={r2}")

            # Saving metrics as local
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)
            print("Json saved.")

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)
            print("Metrics logged to MLflow.")

            print(f"Tracking URL type store: {tracking_url_type_store}")

            # Model registry logic
            try:
                if tracking_url_type_store != "file":
                    # Register the model if not using file store
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetModel")
                    print("Model logged and registered in MLflow.")
                else:
                    mlflow.sklearn.log_model(model, "model")
                    print("Model logged in MLflow.")
            except Exception as e:
                print("An error occurred while logging the model:")
                print(e)

        print("MLflow run completed.")
