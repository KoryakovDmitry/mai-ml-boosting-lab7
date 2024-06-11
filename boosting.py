from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:
    def __init__(
            self,
            base_model_class=DecisionTreeRegressor,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        # Generate a bootstrap sample
        sample_size = int(self.subsample * x.shape[0])
        indices = np.random.choice(x.shape[0], size=sample_size, replace=True)
        x_sample = x[indices]
        y_sample = y[indices]
        residuals = -self.loss_derivative(y_sample, predictions[indices])

        # Train base model
        model = self.base_model_class(**self.base_model_params)
        model.fit(x_sample, residuals)

        # Optimize gamma
        new_predictions = model.predict(x)
        gamma = self.find_optimal_gamma(y, predictions, new_predictions)

        # Update the ensemble
        self.models.append(model)
        self.gammas.append(gamma)
        return gamma, new_predictions

    def fit(self, x_train, y_train, x_valid, y_valid):
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])

        for i in range(self.n_estimators):
            gamma, new_train_predictions = self.fit_new_base_model(x_train, y_train, train_predictions)
            train_predictions += self.learning_rate * gamma * new_train_predictions
            valid_predictions += self.learning_rate * gamma * self.models[-1].predict(x_valid)

            train_loss = self.loss_fn(y_train, train_predictions)
            valid_loss = self.loss_fn(y_valid, valid_predictions)
            self.history['train_loss'].append(train_loss)
            self.history['valid_loss'].append(valid_loss)

            if self.early_stopping_rounds is not None:
                if i >= self.early_stopping_rounds and valid_loss > self.validation_loss.min():
                    print(f"Early stopping on iteration {i + 1}")
                    break
                self.validation_loss[i % self.early_stopping_rounds] = valid_loss

        if self.plot:
            plt.plot(self.history['train_loss'], label='Train Loss')
            plt.plot(self.history['valid_loss'], label='Validation Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

    def predict_proba(self, x):
        predictions = np.zeros(x.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            predictions += self.learning_rate * gamma * model.predict(x)

        probabilities = self.sigmoid(predictions)
        # Return probabilities for both classes
        return np.vstack([1 - probabilities, probabilities]).T

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        total_importances = np.zeros(self.models[0].feature_importances_.shape)
        for model in self.models:
            total_importances += model.feature_importances_
        return total_importances / len(self.models)
