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


ml_manager = MLFlowManager()
# Função para registrar experimento no MLflow
def mlflow_register(model_name="decision_tree_model", stage="Staging", experiment_name="Experimento_ML"):
    ml_manager = MLFlowManager()
    ml_manager.mlflow_register(model_name, stage, experiment_name)
    #ml_manager.start_experiment("Experimento_ML")
    #with ml_manager.start_run():
    print("Experimento registrado no MLflow.")
    #ml_manager.end_run()





# Função para treinar o modelo
def train(X_train, y_train, model = 'decision_tree', params = {'max_depth':5}):
    model = ModelFactory.create_model(model, params)
    model = train_model(model, X_train, y_train)
    print("Passou do treino")
    ml_manager = MLFlowManager()
    print("Conectou com o MLFLOW")
    #ml_manager.start_experiment("Experimento_ML")
    #print("Start no experimento")
    #with ml_manager.start_run():
    ml_manager.log_params({"max_depth": 5})
    print("Modelo treinado e registrado no MLflow.")
    #ml_manager.end_run()
    return model  # retornar o modelo treinado.


# Função para avaliar o modelo
def evaluate(model, X_test, y_test, X_train):
    report = evaluate_model(model, X_test, y_test)
    print("Relatório de Avaliação:")
    print(report)

    ml_manager = MLFlowManager()
    #ml_manager.start_experiment("Experimento_ML")
    #with ml_manager.start_run():
    ml_manager.log_metrics(report["weighted avg"])
    name_model = "decision_tree_model"
    print("Avaliação registrada no MLflow.")
    #ml_manager.end_run()


# Função para realizar inferência
def predict(model, X_predict):
    predictions = predict_model(model, X_predict)
    print("Previsões:")
    print(predictions)

    ml_manager = MLFlowManager()
    #ml_manager.start_experiment("Experimento_ML")
    #with ml_manager.start_run():
    ml_manager.log_text(str(predictions), "predictions.txt")
    print("Previsões registradas no MLflow.")
    #ml_manager.end_run()


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
####### POSSIBILIDADE 1:
    if args.command == "mlflow_register":
        ml_manager.start_experiment('Registro mlflow')
        with ml_manager.start_run():
            mlflow_register()
    elif args.command == "train":
        print("CHEGOU AQUI")
        ml_manager.start_experiment('Treino')
        with ml_manager.start_run():
            model = train(X_train, y_train)
            print("PASSOU AQUI")
    elif args.command == "evaluate":
        ml_manager.start_experiment('Evaluate')
        with ml_manager.start_run():
            model = train(X_train, y_train)  # treina o modelo antes de avaliar.
            evaluate(model, X_test, y_test, X_train)
            name_model = "decision_tree_model"
            ml_manager.log_model(model, name_model, input_example=X_train, signature = model.predict(X_train))
    elif args.command == "predict":
        ml_manager.start_experiment('Predição')
        with ml_manager.start_run():
            model = train(X_train, y_train)  # treina o modelo antes de prever.
            predict(model, X_test)  # usa X_test para prever como exemplo.
            name_model = "decision_tree_model"
            ml_manager.log_model(model, name_model, input_example=X_train, signature=model.predict(X_train))
    print("Foi até o fim")
"""
####### POSSIBILIDADE 2:
    experimento = 'EXPERIMENTO MLFLOW'
    ml_manager.start_experiment(experimento)
    if args.command == "mlflow_register":
        #ml_manager.start_experiment('Registro mlflow')
        with ml_manager.start_run():
            mlflow.set_tag("etapa", "registro")
            mlflow_register()
    elif args.command == "train":
        print("CHEGOU AQUI")
        #ml_manager.start_experiment('Treino')
        with ml_manager.start_run():
            mlflow.set_tag("etapa", "Treino")
            model = train(X_train, y_train)
            print("PASSOU AQUI")
    elif args.command == "evaluate":
        #ml_manager.start_experiment('Evaluate')
        with ml_manager.start_run():
            mlflow.set_tag("etapa", "Evaluate")
            model = train(X_train, y_train)  # treina o modelo antes de avaliar.
            evaluate(model, X_test, y_test)
            ml_manager.log_model(model, name_model, input_example=X_train, signature=model.predict(X_train))
            mlflow_register(model_name=name_model, experiment_name=experimento)
            
    elif args.command == "predict":
        #ml_manager.start_experiment('Predição')
        with ml_manager.start_run():
            mlflow.set_tag("etapa", "Predict")
            model = train(X_train, y_train)  # treina o modelo antes de prever.
            predict(model, X_test)  # usa X_test para prever como exemplo.
            ml_manager.log_model(model, name_model, input_example=X_train, signature=model.predict(X_train))
    print("Foi até o fim")

    # load model que está no mlflow com base em experimento

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Experimento_ML")
    runs = client.search_runs(experiment.experiment_id, order_by=["attributes.start_time DESC"], max_results=1)

    if runs:
        latest_run = runs[0]
        run_id = latest_run.info.run_id
        model_uri = f"runs:/{run_id}/{model_name}"
        loaded_model = mlflow.pyfunc.load_model(model_uri)

    # Load model que está no mlflow que está registrado
    model_name = "DecisionTreeModel"
    stage = "Production"  # ou "Staging", "Archived"

    model_uri = f"models:/{model_name}/{stage}"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
"""

if __name__ == "__main__":
    print("INICIO")
    main()
