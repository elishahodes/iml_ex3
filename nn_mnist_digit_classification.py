import time
import numpy as np
import gzip
from typing import Tuple

from nn_loss_functions import accuracy, softmax
from nn_modules import FullyConnectedLayer, ReLU, CrossEntropyLoss
from neural_network import NeuralNetwork
from gradient_descent import GradientDescent
from learning_rate import FixedLR
from stochastic_gradient_descent import StochasticGradientDescent
from nn_utils import confusion_matrix

import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"

import os
os.makedirs("figures", exist_ok=True)


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def load_images(path):
        with gzip.open(path) as f:
            raw_data = np.frombuffer(f.read(), 'B', offset=16)
        return raw_data.reshape(-1, 784).astype('float32') / 255

    def load_labels(path):
        with gzip.open(path) as f:
            return np.frombuffer(f.read(), 'B', offset=8)

    return (load_images('mnist-train-images.gz'),
            load_labels('mnist-train-labels.gz'),
            load_images('mnist-test-images.gz'),
            load_labels('mnist-test-labels.gz'))


def plot_images_grid(images: np.ndarray, title: str = ""):
    side = int(len(images) ** 0.5)
    subset_images = images.reshape(-1, 28, 28)
    height, width = subset_images.shape[1:]
    grid = subset_images.reshape(side, side, height, width).swapaxes(1, 2).reshape(height * side, width * side)
    return px.imshow(grid, color_continuous_scale="gray")\
        .update_layout(title=dict(text=title, y=0.97, x=0.5, xanchor="center", yanchor="top"),
                       font=dict(size=16), coloraxis_showscale=False)\
        .update_xaxes(showticklabels=False)\
        .update_yaxes(showticklabels=False)


def make_convergence_callback():
    losses, grad_norms = [], []

    def callback(solver, weights, val, grad, t, eta, delta, **kwargs):
        losses.append(float(val[0]))
        grad_norms.append(float(np.linalg.norm(grad)))

    return callback, losses, grad_norms


def make_timed_callback():
    losses, times = [], []
    start = [None]

    def callback(solver, weights, val, grad, t, eta, delta, **kwargs):
        if start[0] is None:
            start[0] = time.time()
        losses.append(float(val[0]))
        times.append(time.time() - start[0])

    return callback, losses, times


def build_two_hidden_nn(n_features, n_classes, width, solver):
    return NeuralNetwork(
        modules=[
            FullyConnectedLayer(n_features, width, activation=ReLU(), include_intercept=True),
            FullyConnectedLayer(width, width, activation=ReLU(), include_intercept=True),
            FullyConnectedLayer(width, n_classes, include_intercept=True),
        ],
        loss_fn=CrossEntropyLoss(),
        solver=solver
    )


if __name__ == '__main__':
    train_X, train_y, test_X, test_y = load_mnist()
    (n_samples, n_features), n_classes = train_X.shape, 10

    # --------------------------------------------------------------------------------------------- #
    # Questions 5+6+7: Network with two ReLU hidden layers (64 neurons) using SGD                  #
    # --------------------------------------------------------------------------------------------- #
    np.random.seed(0)
    callback_q5, losses_q5, grad_norms_q5 = make_convergence_callback()
    nn_q5 = build_two_hidden_nn(
        n_features, n_classes, width=64,
        solver=StochasticGradientDescent(
            learning_rate=FixedLR(0.1), max_iter=10000, tol=1e-10,
            batch_size=256, callback=callback_q5
        )
    )
    nn_q5.fit(train_X, train_y)
    test_acc_q5 = accuracy(test_y, nn_q5.predict(test_X))
    print(f"Q5 test accuracy (SGD, 64 neurons): {test_acc_q5:.4f}")

    # Q6: convergence plots
    iters = list(range(len(losses_q5)))
    fig_q6 = make_subplots(rows=1, cols=2, subplot_titles=("Loss vs Iteration", "Gradient Norm vs Iteration"))
    fig_q6.add_trace(go.Scatter(x=iters, y=losses_q5, mode="lines", name="Loss"), row=1, col=1)
    fig_q6.add_trace(go.Scatter(x=iters, y=grad_norms_q5, mode="lines", name="Grad Norm"), row=1, col=2)
    fig_q6.update_layout(title="Q6: MNIST Convergence (SGD, 64 neurons)", showlegend=False)
    fig_q6.write_image("figures/s222_q6_convergence.png")

    # Q7: confusion matrix
    test_preds_q5 = nn_q5.predict(test_X)
    cm = confusion_matrix(test_y, test_preds_q5)
    labels = [str(i) for i in range(10)]
    fig_cm = ff.create_annotated_heatmap(cm, x=labels, y=labels, colorscale="Blues")
    fig_cm.update_layout(title="Q7: Confusion Matrix", xaxis_title="Predicted", yaxis_title="True")
    fig_cm.write_image("figures/s222_q7_confusion_matrix.png")

    # Find most/least common off-diagonal confusions (only true misclassifications)
    n_cls = cm.shape[0]
    off_diag_vals = [(cm[r, c], r, c) for r in range(n_cls) for c in range(n_cls) if r != c]
    off_diag_vals.sort(key=lambda x: x[0])
    top2 = [(r, c) for _, r, c in off_diag_vals[-2:][::-1]]
    bot3 = [(r, c) for _, r, c in off_diag_vals[:3]]
    print(f"Q7 two most common confusions (true→pred): {top2}")
    print(f"Q7 three least common confusions (true→pred): {bot3}")

    # --------------------------------------------------------------------------------------------- #
    # Question 8: No hidden layers                                                                  #
    # --------------------------------------------------------------------------------------------- #
    np.random.seed(0)
    nn_q8 = NeuralNetwork(
        modules=[FullyConnectedLayer(n_features, n_classes, include_intercept=True)],
        loss_fn=CrossEntropyLoss(),
        solver=StochasticGradientDescent(learning_rate=FixedLR(0.1), max_iter=10000, tol=1e-10, batch_size=256)
    )
    nn_q8.fit(train_X, train_y)
    test_acc_q8 = accuracy(test_y, nn_q8.predict(test_X))
    print(f"Q8 test accuracy (no hidden layers, SGD): {test_acc_q8:.4f}")

    # --------------------------------------------------------------------------------------------- #
    # Question 9: Most/Least confident predictions for digit 7                                     #
    # --------------------------------------------------------------------------------------------- #
    mask_7 = test_y == 7
    X_7 = test_X[mask_7]
    logits_7 = nn_q5.compute_prediction(X_7)
    probs_7 = softmax(logits_7)
    confidence_7 = probs_7.max(axis=1)

    sorted_idx = np.argsort(confidence_7)
    most_conf_idx = sorted_idx[-64:][::-1]
    least_conf_idx = sorted_idx[:64]

    plot_images_grid(X_7[most_conf_idx], title="Q9: 64 Most Confident (digit 7)")\
        .write_image("figures/s222_q9_most_confident.png")
    plot_images_grid(X_7[least_conf_idx], title="Q9: 64 Least Confident (digit 7)")\
        .write_image("figures/s222_q9_least_confident.png")

    # --------------------------------------------------------------------------------------------- #
    # Question 10: GD vs SGD runtime comparison                                                    #
    # --------------------------------------------------------------------------------------------- #
    np.random.seed(0)
    X_small, y_small = train_X[:2500], train_y[:2500]

    # GD
    callback_gd, losses_gd, times_gd = make_timed_callback()
    nn_gd = build_two_hidden_nn(
        n_features, n_classes, width=64,
        solver=GradientDescent(learning_rate=FixedLR(0.1), max_iter=10000, tol=1e-10, callback=callback_gd)
    )
    nn_gd.fit(X_small, y_small)

    # SGD
    callback_sgd, losses_sgd, times_sgd = make_timed_callback()
    nn_sgd = build_two_hidden_nn(
        n_features, n_classes, width=64,
        solver=StochasticGradientDescent(
            learning_rate=FixedLR(0.1), max_iter=10000, tol=1e-10,
            batch_size=64, callback=callback_sgd
        )
    )
    nn_sgd.fit(X_small, y_small)

    # GD plot
    fig_gd = go.Figure(go.Scatter(x=times_gd, y=losses_gd, mode="lines", name="GD"))
    fig_gd.update_layout(title="Q10: GD Runtime vs Loss", xaxis_title="Time (s)", yaxis_title="Loss")
    fig_gd.write_image("figures/s222_q10_gd.png")

    # SGD plot
    fig_sgd_only = go.Figure(go.Scatter(x=times_sgd, y=losses_sgd, mode="lines", name="SGD"))
    fig_sgd_only.update_layout(title="Q10: SGD Runtime vs Loss", xaxis_title="Time (s)", yaxis_title="Loss")
    fig_sgd_only.write_image("figures/s222_q10_sgd.png")

    # Combined
    fig_both = go.Figure([
        go.Scatter(x=times_gd, y=losses_gd, mode="markers", name="GD"),
        go.Scatter(x=times_sgd, y=losses_sgd, mode="markers", name="SGD"),
    ])
    fig_both.update_layout(title="Q10: GD vs SGD Runtime vs Loss", xaxis_title="Time (s)", yaxis_title="Loss")
    fig_both.write_image("figures/s222_q10_combined.png")

    print("Q10 done.")
