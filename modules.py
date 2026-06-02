import numpy as np
from base_module import BaseModule


class L2(BaseModule):
    """
    Class representing the L2 module

    Represents the function: f(w)=||w||^2_2
    """
    def __init__(self, weights: np.ndarray = None):
        super().__init__(weights)

    def compute_output(self, **kwargs) -> np.ndarray:
        return np.array([np.sum(self._weights ** 2)])

    def compute_jacobian(self, **kwargs) -> np.ndarray:
        return 2 * self._weights

    def backprop(self, upstream_grad: np.ndarray) -> np.ndarray:
        pass

    def clear_cache(self) -> None:
        pass


class L1(BaseModule):
    def __init__(self, weights: np.ndarray = None):
        super().__init__(weights)

    def compute_output(self, **kwargs) -> np.ndarray:
        return np.array([np.sum(np.abs(self._weights))])

    def compute_jacobian(self, **kwargs) -> np.ndarray:
        return np.sign(self._weights)

    def backprop(self, upstream_grad: np.ndarray) -> np.ndarray:
        pass

    def clear_cache(self) -> None:
        pass
