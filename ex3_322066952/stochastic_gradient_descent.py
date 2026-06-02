from __future__ import annotations
from typing import Callable, Tuple
import numpy as np

from base_module import BaseModule
from base_learning_rate import BaseLR
from gradient_descent import default_callback
from learning_rate import FixedLR


class StochasticGradientDescent:
    """
    Stochastic Gradient Descent algorithm
    """
    def __init__(self,
                 learning_rate: BaseLR = FixedLR(1e-3),
                 tol: float = 1e-5,
                 max_iter: int = 1000,
                 batch_size: int = 1,
                 callback: Callable[[...], None] = default_callback):
        self._learning_rate = learning_rate
        self._tol = tol
        self._max_iter = max_iter
        self._batch_size = batch_size
        self._callback = callback

    def fit(self, f: BaseModule, X: np.ndarray, y: np.ndarray):
        w = f.weights.copy()

        for t in range(self._max_iter):
            batch_indices = np.random.choice(len(X), self._batch_size, replace=False)
            val, jac, eta = self._partial_fit(f, X[batch_indices], y[batch_indices], t)
            w_new = f.weights
            delta = np.linalg.norm(w_new - w)
            w = w_new

            self._callback(solver=self, weights=w.copy(), val=val, grad=jac,
                           t=t, eta=eta, delta=delta, batch_indices=batch_indices)

            if delta < self._tol:
                break

        return f.weights

    def _partial_fit(self, f: BaseModule, X: np.ndarray, y: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray, float]:
        jac = f.compute_jacobian(X=X, y=y)
        eta = self._learning_rate.lr_step(t=t)
        f.weights = f.weights - eta * jac
        val = f.compute_output(X=X, y=y)
        return val, jac, eta
