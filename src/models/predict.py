def predict_model(model, X_predict):
    """
    Realiza previsões usando o modelo treinado.
    """
    predictions = model.predict(X_predict)
    return predictions
