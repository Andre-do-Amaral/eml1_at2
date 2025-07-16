import mlflow
from mlflow import sklearn
import logging
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient


logging.basicConfig(level=logging.INFO)


class MLFlowManager:
    _instance = None

    def __new__(cls, tracking_uri: str = "sqlite:///mlflow.db"):
        if cls._instance is None:
            cls._instance = super(MLFlowManager, cls).__new__(cls)
            try:
                mlflow.set_tracking_uri(tracking_uri)
                logging.info(f"MLFlowManager initialized with URI: {tracking_uri}")
            except Exception as e:
                logging.error(f"Error initializing MLFlowManager: {e}")
                raise
        return cls._instance
    
    def log_model(self, model, model_name, input_example=None, signature=None):
        logging.info(f"Logging model: {model_name}")
        try:
            if signature is None and input_example is not None:
                signature = infer_signature(input_example)
            sklearn.log_model(model, model_name, signature=signature, input_example=input_example)
            logging.info("Model logged successfully.")
        except Exception as e:
            logging.error(f"Error logging model: {e}")
            raise

    def mlflow_register(model_name="decision_tree_model", stage="Staging", experiment_name="Experimento_ML"):
    
        client = MlflowClient()

        # Pega o experimento
        experiment = client.get_experiment_by_name(experiment_name)
        runs = client.search_runs(experiment.experiment_id, order_by=["attributes.start_time DESC"], max_results=1)

        if not runs:
            print("Nenhum run encontrado.")
            return

        latest_run = runs[0]
        run_id = latest_run.info.run_id
        model_uri = f"runs:/{run_id}/{model_name}"

        # Registra o modelo
        try:
            client.create_registered_model(model_name)
        except:
            pass  # já existe

        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id,
        )

        # Move para o estágio desejado
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage=stage
        )

        print(f"Modelo registrado como '{model_name}' na versão {model_version.version} e movido para o estágio '{stage}'.")

    def start_run(self):
        return mlflow.start_run()
    
    def end_run(self):
        return mlflow.end_run()
    
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

   # def log_model(self, model, model_name):
   #     logging.info(f"Logging model: {model_name}")
   #     try:
   #         sklearn.log_model(model, model_name)
   #         logging.info("Model logged successfully.")
   #     except Exception as e:
   #         logging.error(f"Error logging model: {e}")
   #         raise

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
