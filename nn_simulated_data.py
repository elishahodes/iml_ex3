import numpy as np
import pandas as pd
from typing import Tuple
from nn_presentation_utils import custom, plot_decision_boundary, animate_decision_boundary
from nn_loss_functions import accuracy
from nn_modules import FullyConnectedLayer, ReLU, CrossEntropyLoss
from neural_network import NeuralNetwork
from gradient_descent import GradientDescent
from learning_rate import FixedLR
from nn_utils import split_train_test

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.templates.default = "simple_white"

import os
os.makedirs("figures", exist_ok=True)


def generate_nonlinear_data(
        samples_per_class: int = 100,
        n_features: int = 2,
        n_classes: int = 2,
        train_proportion: float = 0.8) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = np.zeros((samples_per_class*n_classes, n_features)), np.zeros(samples_per_class*n_classes, dtype='uint8')
    for j in range(n_classes):
        ix = range(samples_per_class * j, samples_per_class * (j + 1))
        r = np.linspace(0.0, 1, samples_per_class)
        t = np.linspace(j * 4, (j + 1) * 4, samples_per_class) + np.random.randn(samples_per_class) * 0.2
        X[ix], y[ix] = np.c_[r * np.sin(t), r * np.cos(t)], j

    X_train, y_train, X_test, y_test = split_train_test(pd.DataFrame(X), pd.Series(y), train_proportion=train_proportion)
    return X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()


def make_convergence_callback():
    losses, grad_norms, saved_weights = [], [], []

    def callback(solver, weights, val, grad, t, eta, delta, **kwargs):
        losses.append(float(val[0]))
        grad_norms.append(float(np.linalg.norm(grad)))
        if t % 100 == 0:
            saved_weights.append(weights.copy())

    return callback, losses, grad_norms, saved_weights


if __name__ == '__main__':
    np.random.seed(0)

    n_features, n_classes = 2, 3
    train_X, train_y, test_X, test_y = generate_nonlinear_data(
        samples_per_class=500, n_features=n_features, n_classes=n_classes, train_proportion=0.8)

    lims = ((-1.5, 1.5), (-1.5, 1.5))

    go.Figure(data=[go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode='markers',
                               marker=dict(color=train_y.astype(int), colorscale=custom, line=dict(color="black", width=1)))],
              layout=go.Layout(title=r"$\text{Train Data}$", xaxis=dict(title=r"$x_1$"), yaxis=dict(title=r"$x_2$"),
                               width=400, height=400))\
        .write_image("figures/nonlinear_data.png")

    # --------------------------------------------------------------------------------------------- #
    # Question 1: Two hidden layers, 16 neurons each                                               #
    # --------------------------------------------------------------------------------------------- #
    nn_q1 = NeuralNetwork(
        modules=[
            FullyConnectedLayer(n_features, 16, activation=ReLU(), include_intercept=True),
            FullyConnectedLayer(16, 16, activation=ReLU(), include_intercept=True),
            FullyConnectedLayer(16, n_classes, include_intercept=True),
        ],
        loss_fn=CrossEntropyLoss(),
        solver=GradientDescent(learning_rate=FixedLR(0.1), max_iter=5000, tol=1e-8)
    )
    nn_q1.fit(train_X, train_y)
    test_acc_q1 = accuracy(test_y, nn_q1.predict(test_X))
    print(f"Q1 test accuracy (16 neurons, 2 hidden layers): {test_acc_q1:.4f}")

    plot_decision_boundary(nn_q1, lims, train_X, train_y.astype(int), title="Q1: 2 Hidden Layers (16 neurons)")\
        .write_image("figures/s221_q1_decision_boundary.png")

    # --------------------------------------------------------------------------------------------- #
    # Question 2: No hidden layers                                                                  #
    # --------------------------------------------------------------------------------------------- #
    nn_q2 = NeuralNetwork(
        modules=[
            FullyConnectedLayer(n_features, n_classes, include_intercept=True),
        ],
        loss_fn=CrossEntropyLoss(),
        solver=GradientDescent(learning_rate=FixedLR(0.1), max_iter=5000, tol=1e-8)
    )
    nn_q2.fit(train_X, train_y)
    test_acc_q2 = accuracy(test_y, nn_q2.predict(test_X))
    print(f"Q2 test accuracy (no hidden layers): {test_acc_q2:.4f}")

    plot_decision_boundary(nn_q2, lims, train_X, train_y.astype(int), title="Q2: No Hidden Layers")\
        .write_image("figures/s221_q2_decision_boundary.png")

    # --------------------------------------------------------------------------------------------- #
    # Questions 3+4: Convergence plots and animations                                              #
    # --------------------------------------------------------------------------------------------- #
    for q_num, width in [(3, 16), (4, 6)]:
        callback, losses, grad_norms, saved_w = make_convergence_callback()
        nn_conv = NeuralNetwork(
            modules=[
                FullyConnectedLayer(n_features, width, activation=ReLU(), include_intercept=True),
                FullyConnectedLayer(width, width, activation=ReLU(), include_intercept=True),
                FullyConnectedLayer(width, n_classes, include_intercept=True),
            ],
            loss_fn=CrossEntropyLoss(),
            solver=GradientDescent(learning_rate=FixedLR(0.1), max_iter=5000, tol=1e-8, callback=callback)
        )
        nn_conv.fit(train_X, train_y)

        # Convergence plot
        iters = list(range(len(losses)))
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss vs Iteration", "Gradient Norm vs Iteration"))
        fig.add_trace(go.Scatter(x=iters, y=losses, mode="lines", name="Loss"), row=1, col=1)
        fig.add_trace(go.Scatter(x=iters, y=grad_norms, mode="lines", name="Grad Norm"), row=1, col=2)
        fig.update_layout(title=f"Q{q_num}: Convergence ({width} neurons per layer)", showlegend=False)
        fig.write_image(f"figures/s221_q{q_num}_convergence.png")

        # Animate decision boundary
        animate_decision_boundary(
            nn_conv, saved_w, lims, train_X, train_y.astype(int),
            title=f"Q{q_num}: {width} neurons",
            save_name=f"figures/s221_q{q_num}_animation.gif"
        )
        print(f"Q{q_num} ({width} neurons) done.")
