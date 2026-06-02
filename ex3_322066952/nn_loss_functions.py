import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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
    n = y_pred.shape[0]
    eps = 1e-15
    y_pred_clipped = np.clip(y_pred, eps, 1.0)
    if y_true.ndim == 1:
        log_probs = np.log(y_pred_clipped[np.arange(n), y_true.astype(int)])
    else:
        log_probs = np.sum(y_true * np.log(y_pred_clipped), axis=1)
    return -np.mean(log_probs)


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
    X_shifted = X - X.max(axis=1, keepdims=True)
    exp_X = np.exp(X_shifted)
    return exp_X / exp_X.sum(axis=1, keepdims=True)
