import mlflow
from mlflow import sklearn


class MLFlowManager:
    _instance = None

    def __new__(cls, tracking_uri: str):
        if cls._instance is None:
            cls._instance = super(MLFlowManager, cls).__new__(cls)
            mlflow.set_tracking_uri(tracking_uri)
        return cls._instance

    def start_experiment(self, experiment_name):
        mlflow.set_experiment(experiment_name)

    def log_metrics(self, metrics: dict):
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

    def log_params(self, params: dict):
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_model(self, model, model_name):
        sklearn.log_model(model, model_name)
