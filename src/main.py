from src.mlflow_utils import MLFlowManager
from src.data.preprocess import split_data
from src.models.factory import ModelFactory
from src.models.trainer import train_model
from src.models.evaluator import evaluate_model
from src.data.loaders import load_data


def main():
    ml_manager = MLFlowManager("http://localhost:5000")  # URL do seu servidor MLflow
    ml_manager.start_experiment("Experimento_ML")

    # Carregar e processar dados
    df = load_data("path")
    # Falar a coluna target
    X_train, X_test, X_val, y_train, y_test, y_val = split_data(df, "target")
    # SO TREINO IMPUTA
    # df = impute_missing_values(df)
    # Cria e treina
    model = ModelFactory.create_model("decision_tree", max_depth=5)
    model = train_model(model, X_train, y_train)
    # avaliar modelo
    metrics = evaluate_model(model, X_test, y_test)
    # Regisra
    ml_manager.log_params({"max_depth": 5})
    ml_manager.log_metrics(metrics)
    ml_manager.log_model(model, "decision_tree")


if __name__ == "__main__":
    main()
