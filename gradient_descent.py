from __future__ import annotations
from typing import Callable
import numpy as np

from base_module import BaseModule
from base_learning_rate import BaseLR
from learning_rate import FixedLR

OUTPUT_VECTOR_TYPE = ["last", "best", "average"]


def default_callback(**kwargs) -> None:
    pass


class GradientDescent:
    """
    Gradient Descent algorithm

    Attributes:
    -----------
    _learning_rate: BaseLR
        Learning rate strategy for retrieving the learning rate at each iteration t of the algorithm

    _tol: float
        The stopping criterion. Training stops when the Euclidean norm of w^(t)-w^(t-1) is less than
        specified tolerance

    _max_iter: int
        The maximum number of GD iterations to be performed before stopping training

    _out_type: str
        Type of returned solution:
            - `last`: returns the point reached at the last GD iteration
            - `best`: returns the point achieving the lowest objective
            - `average`: returns the average point over the GD iterations

    _callback: Callable[[...], None], default=default_callback
        A callable function to be called after each update of the model while fitting to given data.
        Callable function receives as input any argument relevant for the current GD iteration. Arguments
        are specified in the `GradientDescent.fit` function
    """
    def __init__(self,
                 learning_rate: BaseLR = FixedLR(1e-3),
                 tol: float = 1e-5,
                 max_iter: int = 1000,
                 out_type: str = "last",
                 callback: Callable[[GradientDescent, ...], None] = default_callback):
        """
        Instantiate a new instance of the GradientDescent class

        Parameters
        ----------
        learning_rate: BaseLR, default=FixedLR(1e-3)
            Learning rate strategy for retrieving the learning rate at each iteration t of the algorithm

        tol: float, default=1e-5
            The stopping criterion. Training stops when the Euclidean norm of w^(t)-w^(t-1) is less than
            specified tolerance

        max_iter: int, default=1000
            The maximum number of GD iterations to be performed before stopping training

        out_type: str, default="last"
            Type of returned solution. Supported types are specified in class attributes

        callback: Callable[[...], None], default=default_callback
            A callable function to be called after each update of the model while fitting to given data.
            Callable function receives as input any argument relevant for the current GD iteration. Arguments
            are specified in the `GradientDescent.fit` function
        """
        self._learning_rate = learning_rate
        if out_type not in OUTPUT_VECTOR_TYPE:
            raise ValueError("output_type not supported")
        self._out_type = out_type
        self._tol = tol
        self._max_iter = max_iter
        self._callback = callback

    def fit(self, f: BaseModule, X: np.ndarray, y: np.ndarray):
        """
        Optimize module using Gradient Descent iterations over given input samples and responses

        Parameters
        ----------
        f : BaseModule
            Module of objective to optimize using GD iterations
        X : ndarray of shape (n_samples, n_features)
            Input data to optimize module over
        y : ndarray of shape (n_samples, )
            Responses of input data to optimize module over

        Returns
        -------
        solution: ndarray of shape (n_features)
            Obtained solution for module optimization, according to the specified self._out_type

        Notes
        -----
        - Optimization is performed as long as self._max_iter has not been reached and that
        Euclidean norm of w^(t)-w^(t-1) is more than the specified self._tol

        - At each iteration the learning rate is specified according to self._learning_rate.lr_step

        - At the end of each iteration the self._callback function is called passing self and the
        following named arguments:
            - solver: GradientDescent
                self, the current instance of GradientDescent
            - weights: ndarray of shape specified by module's weights
                Current weights of objective
            - val: ndarray of shape specified by module's compute_output function
                Value of objective function at current point, over given data X, y
            - grad:  ndarray of shape specified by module's compute_jacobian function
                Module's jacobian with respect to the weights and at current point, over given data X,y
            - t: int
                Current GD iteration
            - eta: float
                Learning rate used at current iteration
            - delta: float
                Euclidean norm of w^(t)-w^(t-1)

        """
        w = f.weights.copy()
        best_val = np.inf
        best_w = w.copy()
        running_sum = w.copy()
        count = 1

        for t in range(self._max_iter):
            grad = f.compute_jacobian(X=X, y=y)
            eta = self._learning_rate.lr_step(t=t)
            w_new = w - eta * grad
            delta = np.linalg.norm(w_new - w)
            w = w_new
            f.weights = w
            val = f.compute_output(X=X, y=y)

            if val[0] < best_val:
                best_val = val[0]
                best_w = w.copy()

            running_sum = running_sum + w
            count += 1

            self._callback(solver=self, weights=w.copy(), val=val, grad=grad, t=t, eta=eta, delta=delta)

            if delta < self._tol:
                break

        if self._out_type == "last":
            return w
        elif self._out_type == "best":
            return best_w
        else:  # average
            return running_sum / count
