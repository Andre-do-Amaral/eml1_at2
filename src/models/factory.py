from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class ModelFactory:
    @staticmethod
    def create_model(model_type: str, **kwargs):
        if model_type == "decision_tree":
            return DecisionTreeClassifier(**kwargs)
        elif model_type == "random_forest":
            return RandomForestClassifier(**kwargs)
        elif model_type == "logistic":
            return LogisticRegression(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
