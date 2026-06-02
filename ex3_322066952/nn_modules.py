import numpy as np
from base_module import BaseModule
from nn_loss_functions import cross_entropy, softmax


class FullyConnectedLayer(BaseModule):
    """
    Module of a fully connected layer in a neural network
    """
    def __init__(self, input_dim: int, output_dim: int, activation: BaseModule = None, include_intercept: bool = True):
        super().__init__()
        self._layer_input_X = None
        self._grad_weights = None
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._activation = activation
        self._include_intercept = include_intercept
        effective_input = input_dim + 1 if include_intercept else input_dim
        self._weights = np.random.normal(0, 1 / input_dim, (effective_input, output_dim))

    def compute_output(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if self._include_intercept:
            ones = np.ones((X.shape[0], 1))
            X_aug = np.concatenate([X, ones], axis=1)
        else:
            X_aug = X
        self._layer_input_X = X_aug
        z = X_aug @ self._weights
        if self._activation is not None:
            return self._activation.compute_output(z)
        return z

    def backprop(self, upstream_grad: np.ndarray) -> np.ndarray:
        if self._activation is not None:
            upstream_grad = self._activation.backprop(upstream_grad)
        self._grad_weights = self._layer_input_X.T @ upstream_grad
        downstream = upstream_grad @ self._weights.T
        if self._include_intercept:
            return downstream[:, :-1]
        return downstream

    def get_grad_weights(self) -> np.ndarray:
        return self._grad_weights

    def clear_cache(self) -> None:
        self._layer_input_X = None
        self._grad_weights = None
        if self._activation is not None:
            self._activation.clear_cache()


class ReLU(BaseModule):
    """
    Module of a ReLU activation function computing the element-wise function ReLU(x)=max(x,0)
    """
    def __init__(self):
        super().__init__()
        self._activation_input_X = None

    def compute_output(self, X: np.ndarray, **kwargs) -> np.ndarray:
        self._activation_input_X = X
        return np.maximum(0, X)

    def backprop(self, upstream_grad: np.ndarray) -> np.ndarray:
        return upstream_grad * (self._activation_input_X > 0)

    def clear_cache(self) -> None:
        self._activation_input_X = None


class CrossEntropyLoss(BaseModule):
    """
    Module of Cross-Entropy Loss
    """
    def compute_output(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        return np.array([cross_entropy(y, softmax(X))])

    def compute_jacobian(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n, k = X.shape
        probs = softmax(X)
        if y.ndim == 1:
            Y_onehot = np.zeros((n, k))
            Y_onehot[np.arange(n), y.astype(int)] = 1
        else:
            Y_onehot = y
        return (probs - Y_onehot) / n

    def backprop(self, upstream_grad: np.ndarray) -> np.ndarray:
        pass

    def clear_cache(self) -> None:
        pass
