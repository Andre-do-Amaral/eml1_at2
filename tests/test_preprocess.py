import pandas as pd
import numpy as np
from src.data.preprocess import impute_missing_values, remove_outliers, split_data


def test_impute_missing_values():
    # Dados fictícios com valores ausentes (treino)
    train_data = pd.DataFrame({"feature1": [1, 2, np.nan], "feature2": [4, np.nan, 6]})

    # Dados fictícios para teste (não deve ter valores ausentes após transformar)
    test_data = pd.DataFrame({"feature1": [3, np.nan, 5], "feature2": [np.nan, 8, 9]})

    # Aplicar a imputação
    imputer = impute_missing_values(
        train_data.copy()
    )  # Passa uma copia para nao modificar o original.
    train_data = pd.DataFrame(imputer.transform(train_data), columns=train_data.columns)

    # Verificar se os valores ausentes no treino foram preenchidos
    assert not train_data.isnull().any().any()

    # Transformar os dados de teste
    transformed_test_data = pd.DataFrame(
        imputer.transform(test_data), columns=test_data.columns
    )

    # Verificar se os valores ausentes no teste foram preenchidos
    assert not transformed_test_data.isnull().any().any()


def test_remove_outliers():
    # Dados fictícios com outliers
    data = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 100],  # 100 é um outlier
            "feature2": [4, 5, 6, 7],
        }
    )

    # Remover outliers
    cleaned_data = remove_outliers(data, ["feature1"])

    # Verificar se o outlier foi removido
    assert cleaned_data["feature1"].max() < 100


def test_split_data():
    # Dados fictícios
    data = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 10, 5, 7, 5, 4, 3, 5, 6],
            "feature2": [5, 6, 7, 8, 10, 5, 7, 5, 4, 3, 5, 6],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    # Dividir os dados
    X_train, X_test, X_val, y_train, y_test, y_val = split_data(data, "target")

    # Verificar tamanhos das divisões
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_val) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0
    assert len(y_val) > 0
