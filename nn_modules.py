import numpy as np
from base_module import BaseModule
from nn_loss_functions import cross_entropy, softmax


class FullyConnectedLayer(BaseModule):
    """
    Module of a fully connected layer in a neural network

    Attributes:
    -----------
    _input_dim: int
        Size of input to layer (number of neurons in preceding layer)

    _output_dim: int
        Size of layer output (number of neurons in layer)

    _activation: BaseModule
        Activation function to be performed after integration of inputs and weights

    _weights: ndarray of shape (_input_dim, _output_dim)
        Parameters of function with respect to which the function is optimized.

    _include_intercept: bool
        Should layer include an intercept or not

    _grad_weights : np.ndarray of shape (_input_dim, _output_dim)
        Accumulated gradient of the loss with respect to _weights.
        Calculated during the backprop() call.

    _layer_input_X : np.ndarray of shape (n_samples, _input_dim)
        The input data from the forward pass, potentially augmented with
        a column of ones. Used to calculate the weight gradient.
    """
    def __init__(self, input_dim: int, output_dim: int, activation: BaseModule = None, include_intercept: bool = True):
        """
        Initialize a module of a fully connected layer

        Parameters:
        -----------
        input_dim: int
            Size of input to layer (number of neurons in preceding layer)

        output_dim: int
            Size of layer output (number of neurons in layer)

        activation: BaseModule, default=None
            Activation function to be performed after integration of inputs and weights. If
            none is specified functions as a linear layer

        include_intercept: bool, default=True
            Should layer include an intercept or not

        Notes:
        ------
        Weights are randomly initialized following N(0, 1/input_dim)
        """
        super().__init__()
        self._layer_input_X = None
        self._grad_weights = None
        raise NotImplementedError()

    def compute_output(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute activation(weights @ x) for every sample x (with intercept if
        include_intercept), and store in self._layer_input_X the input of the
        layer for the backward pass.

        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            Input data to be integrated with weights

        Returns:
        --------
        output: ndarray of shape (n_samples, output_dim)
            Value of function at point self._weights
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
        You should calculate and store the exact gradient of the loss with
        respect to the weights in `self._grad_weights`: ndarray of shape (input_dim, output_dim).
        """
        raise NotImplementedError()

    def get_grad_weights(self) -> np.ndarray:
        """
        Retrieve the gradients of the loss with respect to the layer's weights.
        This method returns the gradients calculated during the most recent
        call to `backprop`.

        Returns
        -------
        grad_weights : ndarray of shape (input_dim, output_dim)
            The accumulated gradients for the weight matrix, including the
            intercept/bias term if `include_intercept` is True.

        Notes
        -----
        The value returned by this method is only valid after a forward pass
        and a subsequent backward pass have been performed.
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


class ReLU(BaseModule):
    """
    Module of a ReLU activation function computing the element-wise function ReLU(x)=max(x,0)

     Attributes:
    -----------
    _activation_input_X: np.ndarray of shape (n_samples, input_dim)
    The input data from the forward pass
    """
    def __init__(self):
        super().__init__()
        self._activation_input_X = None

    def compute_output(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute element-wise value of activation, and store the input in
        self._activation_input_X for further use in backward pass.

        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            Input data to be passed through activation

        Returns:
        --------
        output: ndarray of shape (n_samples, input_dim)
            Data after performing the ReLU activation function
        """
        raise NotImplementedError()

    def backprop(self, upstream_grad: np.ndarray) -> np.ndarray:
        """
        Perform a backward pass through the module using the chain rule.

        Receives an upstream matrix of partial derivatives and computes the
        downstream matrix to be passed to the preceding layer.

        Parameters
        ----------
        upstream_grad : ndarray of shape (n_samples, input_dim)
            The matrix of partial derivatives from the succeeding layer.

        Returns
        -------
        downstream_grad : ndarray of shape (n_samples, input_dim)
            The matrix of partial derivatives with respect to the module's
            input, shaped for the next step in the backward chain.
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


class CrossEntropyLoss(BaseModule):
    """
    Module of Cross-Entropy Loss: The Cross-Entropy between the Softmax of a sample x and e_k for a true class k
    """
    def compute_output(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the average Cross-Entropy loss over the batch.

        Calculates the Softmax of the input, computes the Cross-Entropy for
        each sample relative to its true class, and returns the average loss.

        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            The raw logits (pre-softmax scores) from the final layer.

        y: ndarray of shape (n_samples,) or (n_samples, input_dim)
            True labels. Can be categorical indices or one-hot encoded vectors.

        Returns:
        --------
        output: ndarray of shape (1, )
            The average Cross-Entropy loss across all samples in the batch.
        """
        raise NotImplementedError()

    def compute_jacobian(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes the partial derivatives of the average Cross-Entropy loss
        with respect to the input logits.

        Parameters:
        -----------
        X: ndarray of shape (n_samples, input_dim)
            The raw logits (pre-softmax scores) from the final layer.

        y: ndarray of shape (n_samples,) or (n_samples, input_dim)
            True labels. Can be categorical indices or one-hot encoded vectors.

        Returns:
        --------
        output: ndarray of shape (n_samples, input_dim)
            A matrix of partial derivatives shaped to be compatible with
            the subsequent backward pass.
        """
        raise NotImplementedError()
