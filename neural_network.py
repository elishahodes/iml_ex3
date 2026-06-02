import numpy as np
from typing import List, Union
from base_module import BaseModule
from base_estimator import BaseEstimator
from stochastic_gradient_descent import StochasticGradientDescent
from gradient_descent import GradientDescent
from nn_modules import FullyConnectedLayer


class NeuralNetwork(BaseEstimator, BaseModule):
    """
    Class representing a feed-forward fully-connected neural network
    """
    def __init__(self,
                 modules: List[FullyConnectedLayer],
                 loss_fn: BaseModule,
                 solver: Union[StochasticGradientDescent, GradientDescent]):
        super().__init__()
        self._forward_pass_result = None
        self._modules = modules
        self._loss_fn = loss_fn
        self._solver = solver

    # region BaseEstimator implementations
    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._solver.fit(self, X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self.compute_prediction(X=X).argmax(axis=-1)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.compute_output(X, y)[0]
    # endregion

    # region BaseModule implementations
    def compute_output(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        h = X
        for module in self._modules:
            h = module.compute_output(h)
        self._forward_pass_result = h
        return self._loss_fn.compute_output(h, y)

    def compute_prediction(self, X: np.ndarray):
        h = X
        for module in self._modules:
            h = module.compute_output(h)
        return h

    def compute_jacobian(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        self.compute_output(X, y)
        grad = self._loss_fn.compute_jacobian(self._forward_pass_result, y)
        grad_weights_list = [None] * len(self._modules)
        for i, module in enumerate(reversed(self._modules)):
            grad = module.backprop(grad)
            grad_weights_list[len(self._modules) - 1 - i] = module.get_grad_weights()
            module.clear_cache()
        return NeuralNetwork._flatten_parameters(grad_weights_list)

    def clear_cache(self) -> None:
        for module in self._modules:
            module.clear_cache()

    @property
    def weights(self) -> np.ndarray:
        return NeuralNetwork._flatten_parameters([module.weights for module in self._modules])

    @weights.setter
    def weights(self, weights) -> None:
        non_flat_weights = NeuralNetwork._unflatten_parameters(weights, self._modules)
        for module, weights in zip(self._modules, non_flat_weights):
            module.weights = weights
    # endregion

    # region Internal methods
    @staticmethod
    def _flatten_parameters(params: List[np.ndarray]) -> np.ndarray:
        return np.concatenate([grad.flatten() for grad in params])

    @staticmethod
    def _unflatten_parameters(flat_params: np.ndarray, modules: List[BaseModule]) -> List[np.ndarray]:
        low, param_list = 0, []
        for module in modules:
            r, c = module.shape
            high = low + r * c
            param_list.append(flat_params[low: high].reshape(module.shape))
            low = high
        return param_list
    # endregion
