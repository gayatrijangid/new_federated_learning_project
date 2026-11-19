import numpy as np

class LocalLogisticRegression:
    """
    sigma(z) = 1 / (1 + e^-z)
    y_hat = sigma(X_b . w),  X_b = [X | 1]  (bias column appended)
    grad_w = (1/n) X_b^T (y_hat - y)
    w <- w - eta * grad_w
    """
    def __init__(self, learning_rate: float = 0.1, epochs: int = 5):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -25, 25)))

    def train(self, X: np.ndarray, y: np.ndarray, initial_weights: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        X_b = np.hstack([X, np.ones((n_samples, 1))])
        w = initial_weights.copy()

        for _ in range(self.epochs):
            y_hat = self.sigmoid(X_b @ w)
            grad = (1.0 / n_samples) * (X_b.T @ (y_hat - y))
            w -= self.learning_rate * grad

        return w

    def predict_proba(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        X_b = np.hstack([X, np.ones((n_samples, 1))])
        return self.sigmoid(X_b @ weights)

    def predict(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X, weights) >= 0.5).astype(int)
