from sklearn.tree import DecisionTreeClassifier
from src.models.trainer import train_model


def test_train_model():
    # Dados fictícios para treino
    X_train = [[1, 2], [3, 4], [5, 6]]
    y_train = [0, 1, 0]

    # Modelo
    model = DecisionTreeClassifier(max_depth=2)

    # Treinamento
    trained_model = train_model(model, X_train, y_train)

    # Verificar se o modelo foi treinado
    assert trained_model is not None
    assert hasattr(
        trained_model, "predict"
    )  # O modelo treinado deve ser capaz de fazer previsões
