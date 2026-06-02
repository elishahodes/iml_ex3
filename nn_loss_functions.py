import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Accuracy of given predictions
    """
    score = np.sum(y_true == y_pred)
    return score / len(y_true)


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the cross entropy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples,) or (n_samples, input_dim)
            True labels. Can be categorical indices or one-hot encoded vectors.
    y_pred: ndarray of shape (n_samples, input_dim)
        Predicted distribution (softmax) of each sample

    Returns
    -------
    output: float
        Cross entropy of given y_true, y_pred
    """
    raise NotImplementedError()


def softmax(X: np.ndarray) -> np.ndarray:
    """
    Compute the Softmax function for each sample in given data

    Parameters:
    -----------
    X: ndarray of shape (n_samples, input_dim)

    Returns:
    --------
    output: ndarray of shape (n_samples, input_dim)
        Softmax(x) for every sample x in the given data X
    """
    raise NotImplementedError()
