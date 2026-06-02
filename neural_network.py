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

    Attributes:
    ----------
    _modules: List[FullyConnectedLayer]
        A list of network layers, each a fully connected layer with its specified activation function

    _loss_fn: BaseModule
        Network's loss function to optimize weights with respect to

    _solver: Union[StochasticGradientDescent, GradientDescent]
        Instance of optimization algorithm used to optimize network
    """
    def __init__(self,
                 modules: List[FullyConnectedLayer],
                 loss_fn: BaseModule,
                 solver: Union[StochasticGradientDescent, GradientDescent]):
        super().__init__()
        self._forward_pass_result = None
        self._modules = modules
        raise NotImplementedError()

    # region BaseEstimator implementations
    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit network over given input data using specified architecture and solver

        Parameters
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        raise NotImplementedError()

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for given samples using fitted network

        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted labels of given samples
        """
        return self.compute_prediction(X=X).argmax(axis=-1)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates network's loss over given data

        Parameters
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        --------
        loss : float
            Performance under specified loss function
        """
        raise NotImplementedError()
    # endregion

    # region BaseModule implementations
    def compute_output(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute network output with respect to modules' weights given input samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        output: ndarray of shape (1, )
            Network's output value including pass through the specified loss function
        """
        raise NotImplementedError()

    def compute_prediction(self, X: np.ndarray):
        """
        Compute network output (forward pass) with respect to modules' weights given input samples, except pass
        through specified loss function (and softmax)

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        Returns
        -------
        output : ndarray of shape (n_samples, n_classes)
            Network's output values prior to the call of the loss function
        """
        raise NotImplementedError()

    def compute_jacobian(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform the network's backward pass using the backpropagation algorithm.

        This method executes the chain rule across the entire network by
        traversing the layers in reverse order. It aggregates the resulting
        gradients of the scalar loss with respect to all trainable parameters.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data for the forward pass.
        y : ndarray of shape (n_samples, )
            Ground-truth labels used to compute the initial loss derivative.

        Returns
        -------
        output : ndarray
            A single flattened vector containing the gradients of the loss with
            respect to every parameter, concatenated in the order of the architecture.
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
        Get flattened weights vector representing the entire network state.

        Returns
        --------
        weights : ndarray of shape (total_params,)
            A 1D vector containing all trainable parameters (weights and intercepts)
            from all layers, concatenated in order.
        """
        return NeuralNetwork._flatten_parameters([module.weights for module in self._modules])

    @weights.setter
    def weights(self, weights) -> None:
        """
        Update the network's parameters from a flattened representation.

        This method is used by solvers (optimizers) to push updated values back into
        the network. It partitions the 1D input vector back into the original
        multidimensional shapes required by each layer, including intercept terms.

        Parameters:
        -----------
        weights : np.ndarray of shape (n_params,)
            A flat vector containing the concatenated weights and intercepts of
            all layers in the network.
        """
        non_flat_weights = NeuralNetwork._unflatten_parameters(weights, self._modules)
        for module, weights in zip(self._modules, non_flat_weights):
            module.weights = weights
    # endregion

    # region Internal methods
    @staticmethod
    def _flatten_parameters(params: List[np.ndarray]) -> np.ndarray:
        """
        Flattens list of all given weights to a single one dimensional vector. To be used when passing
        weights to the solver

        Parameters
        ----------
        params : List[np.ndarray]
            List of differently shaped weight matrices

        Returns
        -------
        weights: ndarray
            A flattened array containing all weights
        """
        return np.concatenate([grad.flatten() for grad in params])

    @staticmethod
    def _unflatten_parameters(flat_params: np.ndarray, modules: List[BaseModule]) -> List[np.ndarray]:
        """
        Performing the inverse operation of "flatten_parameters"

        Parameters
        ----------
        flat_params : ndarray of shape (n_weights,)
            A flat vector containing all weights

        modules : List[BaseModule]
            List of network layers to be used for specifying shapes of weight matrices

        Returns
        -------
        weights: List[ndarray]
            A list where each item contains the weights of the corresponding layer of the network, shaped
            as expected by layer's module
        """
        low, param_list = 0, []
        for module in modules:
            r, c = module.shape
            high = low + r * c
            param_list.append(flat_params[low: high].reshape(module.shape))
            low = high
        return param_list
    # endregion