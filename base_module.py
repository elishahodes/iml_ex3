from __future__ import annotations
from abc import ABC
from typing import Tuple
import numpy as np


class BaseModule(ABC):
    """
    Base class representing a function to be optimized in a descent method algorithm

    Attributes
    ----------
    _weights : ndarray of shape (n_in, n_out)
        Parameters of function with respect to which the function is optimized.
    """

    def __init__(self, weights: np.ndarray = None):
        """
        Initialize a module instance

        Parameters:
        ----------
        weights: np.ndarray, default None
            Initial value of weights
        """
        self._weights = weights

    def compute_output(self, **kwargs) -> np.ndarray:
        """
        Compute the output value of the function

        Parameters
        ----------
        kwargs: Additional arguments to be passed and used by derived objects

        Returns
        -------
        output: ndarray of shape (n_out,)
            Value of function at `input`

        Examples
        --------
        For f:R^d->R defined by f(x) = <w,x> then: n_in=d, n_out=1 and thus output shape is (1,)
        """
        raise NotImplementedError()

    def backprop(self, upstream_grad: np.ndarray) -> np.ndarray:
        """
        Perform a backward pass through the module using the chain rule.

        Receives an upstream matrix of partial derivatives and computes the
        downstream matrix to be passed to the preceding layer.

        Parameters
        ----------
        upstream_grad : ndarray of shape (n_samples, output_dim)
            The matrix of partial derivatives from the succeeding layer.

        Returns
        -------
        downstream_grad : ndarray of shape (n_samples, input_dim)
            The matrix of partial derivatives with respect to the module's
            input, shaped for the next step in the backward chain.

        Notes
        -----
        If the module has trainable parameters, this method must also calculate
        the exact gradient with respect to them and store it in `self._grad_weights`.
        """
        raise NotImplementedError()

    def compute_jacobian(self, **kwargs) -> np.ndarray:
        """
        Compute the partial derivatives of the scalar loss with respect to
        the inputs or parameters, with the requested shape.

        Parameters
        ----------
        **kwargs: Context-specific arguments required for the derivative
                 calculation (e.g., input data X or ground-truth labels y).

        Returns
        -------
        output: ndarray
            The derivative representation, with the requested shap.
        """
        raise NotImplementedError()

    def clear_cache(self) -> None:
        """
        Deletes cached tensors from the forward and backward passes.

        Clearing inputs, pre-activations, and gradients prevents memory accumulation
        and OOM errors. This should be called only after the optimizer has
        collected the gradients.
        """
        raise NotImplementedError()

    @property
    def weights(self) -> np.ndarray:
        """
        Wrapper property to retrieve module parameter

        Returns
        -------
        weights: ndarray of shape (n_in, n_out)
        """
        return self._weights

    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        """
        Setter function for module parameters

        Parameters
        ----------
        weights: ndarray array of shape (n_in, n_out)
        """
        self._weights = weights

    @property
    def shape(self) -> Tuple[int]:
        """
        Specify the dimensions of the function

        Returns
        -------
        shape: Tuple[int]
            Specifying the dimensions of the functions parameters. If ``self.weights`` is None returns `(0,)`
        """
        return self.weights.shape if self.weights is not None else (0,)


