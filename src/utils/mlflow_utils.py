import mlflow
from mlflow import sklearn
import logging

logging.basicConfig(level=logging.INFO)


class MLFlowManager:
    _instance = None

    def __new__(cls, tracking_uri: str):
        if cls._instance is None:
            cls._instance = super(MLFlowManager, cls).__new__(cls)
            try:
                mlflow.set_tracking_uri(tracking_uri)
                logging.info(f"MLFlowManager initialized with URI: {tracking_uri}")
            except Exception as e:
                logging.error(f"Error initializing MLFlowManager: {e}")
                raise
        return cls._instance

    def start_experiment(self, experiment_name):
        logging.info(f"Starting experiment: {experiment_name}")
        try:
            mlflow.set_experiment(experiment_name)
            logging.info(f"Experiment '{experiment_name}' started successfully.")
        except Exception as e:
            logging.error(f"Error starting experiment: {e}")
            raise

    def log_metrics(self, metrics: dict):
        logging.info(f"Logging metrics: {metrics}")
        try:
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            logging.info("Metrics logged successfully.")
        except Exception as e:
            logging.error(f"Error logging metrics: {e}")
            raise

    def log_params(self, params: dict):
        logging.info(f"Logging params: {params}")
        try:
            for key, value in params.items():
                mlflow.log_param(key, value)
            logging.info("Params logged successfully.")
        except Exception as e:
            logging.error(f"Error logging params: {e}")
            raise

    def log_model(self, model, model_name):
        logging.info(f"Logging model: {model_name}")
        try:
            sklearn.log_model(model, model_name)
            logging.info("Model logged successfully.")
        except Exception as e:
            logging.error(f"Error logging model: {e}")
            raise

    def check_connection(self):
        logging.info("Checking MLflow connection...")
        try:
            mlflow.get_experiment_by_name("Default")
            logging.info("MLflow connection is working.")
            return True
        except Exception as e:
            logging.error(f"MLflow connection error: {e}")
            return False

    def log_text(self, text: str, artifact_file: str):
        logging.info(f"Logging text to {artifact_file}")
        try:
            mlflow.log_text(text, artifact_file)
            logging.info(f"Text logged to {artifact_file} successfully.")
        except Exception as e:
            logging.error(f"Error logging text: {e}")
            raise
