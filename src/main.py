from utils.mlflow_utils import MLFlowManager
from models.factory import ModelFactory
from models.trainer import train_model
from data.preprocess import split_data, impute_missing_values
from data.loaders import load_data
from models.predict import predict_model
from models.evaluator import evaluate_model
import argparse
import pandas as pd
import mlflow
import os


# Função para registrar experimento no MLflow
def mlflow_register():
    ml_manager = MLFlowManager()
    ml_manager.start_experiment("Experimento_ML")
    print("Experimento registrado no MLflow.")
    mlflow.end_run()


# Função para treinar o modelo
def train(X_train, y_train):
    model = ModelFactory.create_model("decision_tree", max_depth=5)
    model = train_model(model, X_train, y_train)
    print("Passou do treino")
    ml_manager = MLFlowManager()
    print("Conectou com o MLFLOW")
    ml_manager.start_experiment("Experimento_ML")
    print("Start no experimento")
    ml_manager.log_params({"max_depth": 5})
    print("Modelo treinado e registrado no MLflow.")
    mlflow.end_run()
    return model  # retornar o modelo treinado.


# Função para avaliar o modelo
def evaluate(model, X_test, y_test):
    report = evaluate_model(model, X_test, y_test)
    print("Relatório de Avaliação:")
    print(report)

    ml_manager = MLFlowManager()
    ml_manager.start_experiment("Experimento_ML")
    ml_manager.log_metrics(report["weighted avg"])
    print("Avaliação registrada no MLflow.")
    mlflow.end_run()


# Função para realizar inferência
def predict(model, X_predict):
    predictions = predict_model(model, X_predict)
    print("Previsões:")
    print(predictions)

    ml_manager = MLFlowManager()
    ml_manager.start_experiment("Experimento_ML")
    ml_manager.log_text(str(predictions), "predictions.txt")
    print("Previsões registradas no MLflow.")
    mlflow.end_run()


# CLI para gerenciar os comandos
def main():
    print("Caminho atual:", os.getcwd())
    df = load_data("/root/MEU_PROJETO/src/water_potability.csv")  # caminho
    X_train, X_test, X_val, y_train, y_test, y_val = split_data(
        df, "Potability"
    )  # target

    imputer = impute_missing_values(X_train.copy())
    X_train = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    X_val = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)

    parser = argparse.ArgumentParser(
        description="Gerenciar modelos de Machine Learning"
    )
    parser.add_argument(
        "command",
        choices=["mlflow_register", "train", "evaluate", "predict"],
        help="Comando a ser executado",
    )
    print("Dados Pre-processados")
    args = parser.parse_args()

    if args.command == "mlflow_register":
        mlflow_register()
    elif args.command == "train":
        print("CHEGOU AQUI")
        model = train(X_train, y_train)
        print("PASSOU AQUI")
    elif args.command == "evaluate":
        model = train(X_train, y_train)  # treina o modelo antes de avaliar.
        evaluate(model, X_test, y_test)
    elif args.command == "predict":
        model = train(X_train, y_train)  # treina o modelo antes de prever.
        predict(model, X_test)  # usa X_test para prever como exemplo.
    print("Foi até o fim")


if __name__ == "__main__":
    print("INICIO")
    main()
