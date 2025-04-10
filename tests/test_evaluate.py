from sklearn.tree import DecisionTreeClassifier
from src.models.evaluator import evaluate_model


def test_evaluate_model():
    # Dados fictícios
    X_test = [[1, 2], [3, 4]]
    y_test = [0, 1]

    # Modelo fictício
    model = DecisionTreeClassifier()
    model.fit(X_test, y_test)

    # Avaliação
    report = evaluate_model(model, X_test, y_test)

    # Verificar se as métricas foram calculadas corretamente
    assert (
        "accuracy" in report
    )  # Verifica se a métrica de acurácia está presente no relatório
    assert isinstance(report["accuracy"], float)  # A métrica deve ser um número
