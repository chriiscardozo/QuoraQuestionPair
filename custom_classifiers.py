import random
from sklearn.metrics import log_loss

class RandomClassifier:
    def fit(self, X, y):
        return

    def _generate_y(self, n):
        y = []
        for i in range(n):
            y.append(random.random())
        return y

    def predict_proba(self, X):
        return self._generate_y(len(X))

class SimpleClassClassifier:
    def __init__(self, value):
        self.v = value

    def fit(self, X, y):
        return

    def _generate_y(self, n):
        y = []
        for i in range(n):
            y.append(self.v)
        return y

    def predict_proba(self, X):
        return self._generate_y(len(X))
