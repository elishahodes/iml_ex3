import numpy as np
from typing import Tuple, List, Callable, Type

from base_module import BaseModule
from modules import L1, L2
from gradient_descent import GradientDescent
from learning_rate import FixedLR, ExponentialLR

import plotly.graph_objects as go
import os


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from nn_presentation_utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    recorded_weights = []

    def callback(solver, weights, val, grad, t, eta, delta, **kwargs):
        values.append(val.copy())
        recorded_weights.append(weights.copy())

    return callback, values, recorded_weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    os.makedirs("figures", exist_ok=True)

    for module_class in [L1, L2]:
        name = module_class.__name__

        all_weights = {}
        all_values = {}

        # Run GD for each eta
        for eta in etas:
            callback, values, rec_weights = get_gd_state_recorder_callback()
            module = module_class(weights=init.copy())
            gd = GradientDescent(learning_rate=FixedLR(eta), max_iter=1000, tol=1e-8, callback=callback)
            gd.fit(module, None, None)
            all_weights[eta] = rec_weights
            all_values[eta] = values

        # Plot descent path for each eta
        for eta in etas:
            path = np.array(all_weights[eta])
            fig = plot_descent_path(module_class, path, title=f"{name} eta={eta}")
            fig.write_image(f"figures/{name}_descent_path_eta_{eta}.png")

        # Plot convergence (norm vs iteration) for all etas
        conv_fig = go.Figure()
        for eta in etas:
            norms = [np.linalg.norm(w) for w in all_weights[eta]]
            conv_fig.add_trace(go.Scatter(y=norms, mode="lines", name=f"eta={eta}"))
        conv_fig.update_layout(
            title=f"{name}: Convergence Rate (||w|| vs Iteration)",
            xaxis_title="Iteration",
            yaxis_title="||w||"
        )
        conv_fig.write_image(f"figures/{name}_convergence_fixed_lr.png")

        # Report lowest loss
        for eta in etas:
            losses = [v[0] for v in all_values[eta]]
            print(f"{name} eta={eta}: lowest loss = {min(losses):.6f}")


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    os.makedirs("figures", exist_ok=True)

    all_weights = {}
    all_values = {}

    for gamma in gammas:
        callback, values, rec_weights = get_gd_state_recorder_callback()
        module = L1(weights=init.copy())
        gd = GradientDescent(learning_rate=ExponentialLR(eta, gamma), max_iter=1000, tol=1e-8, callback=callback)
        gd.fit(module, None, None)
        all_weights[gamma] = rec_weights
        all_values[gamma] = values

    # Plot convergence for all gammas
    conv_fig = go.Figure()
    for gamma in gammas:
        norms = [np.linalg.norm(w) for w in all_weights[gamma]]
        conv_fig.add_trace(go.Scatter(y=norms, mode="lines", name=f"gamma={gamma}"))
    conv_fig.update_layout(
        title="L1: Convergence Rate with Exponential Decay (||w|| vs Iteration)",
        xaxis_title="Iteration",
        yaxis_title="||w||"
    )
    conv_fig.write_image("figures/L1_convergence_exp_decay.png")

    # Report lowest loss for each gamma
    for gamma in gammas:
        losses = [v[0] for v in all_values[gamma]]
        print(f"L1 gamma={gamma}: lowest loss = {min(losses):.6f}")

    # Plot descent path for gamma=0.95
    path = np.array(all_weights[0.95])
    fig = plot_descent_path(L1, path, title=f"L1 ExponentialLR eta={eta} gamma=0.95")
    fig.write_image("figures/L1_descent_path_exp_decay_gamma_095.png")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
