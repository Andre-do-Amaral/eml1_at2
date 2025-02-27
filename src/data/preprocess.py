import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split


def impute_missing_values(df: pd.DataFrame, n_neighbors: int = 4) -> KNNImputer:
    """
    Imputa valores ausentes usando o KNNImputer.
    Retorna o objeto do imputador para aplicar transformações futuras.
    """
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    knn_imputer.fit(df)  # Treinar o imputador
    return knn_imputer


def remove_outliers(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df


def split_data(df, target_column, test_size=0.3, val_size=0.33):
    x = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_aux, y_train, y_aux = train_test_split(
        x, y, stratify=y, test_size=test_size
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_aux, y_aux, test_size=val_size, stratify=y_aux
    )
    return X_train, X_test, X_val, y_train, y_test, y_val
