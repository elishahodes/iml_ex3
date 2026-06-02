# Basic imports and settings for working with data
import numpy as np
from typing import List
import os

# Imports and settings for plotting of graphs
import plotly.io as pio
import plotly.graph_objects as go
import imageio.v3 as iio

# Local imports
from neural_network import NeuralNetwork


pio.templates["custom"] = go.layout.Template(
    layout=go.Layout(
        margin=dict(l=20, r=20, t=40, b=0)
    )
)
pio.templates.default = "simple_white+custom"


class AnimationButtons():
    @staticmethod
    def play_scatter(frame_duration = 500, transition_duration = 300):
        return dict(label="Play", method="animate", args=
                    [None, {"frame": {"duration": frame_duration, "redraw": False},
                            "fromcurrent": True, "transition": {"duration": transition_duration, "easing": "quadratic-in-out"}}])

    @staticmethod
    def play(frame_duration = 100, transition_duration = 0):
        return dict(label="Play", method="animate", args=
                    [None, {"frame": {"duration": frame_duration, "redraw": True},
                            "mode":"immediate",
                            "fromcurrent": True, "transition": {"duration": transition_duration, "easing": "linear"}}])

    @staticmethod
    def pause():
        return dict(label="Pause", method="animate", args=
                    [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])

    @staticmethod
    def slider(frame_names):       
        steps= [dict(args=[[i], dict(frame={'duration': 300, 'redraw': False}, mode="immediate", transition= {'duration': 300})],
                           label=i+1, method="animate")
                for i, n in enumerate(frame_names)]
        
        return [dict(yanchor="top", xanchor="left",
                     currentvalue={'font': {'size': 16}, 'prefix': 'Frame: ', 'visible': True, 'xanchor': 'right'},
                     transition={'duration': 0, 'easing': 'linear'},
                     pad= {'b': 10, 't': 50},
                     len=0.9, x=0.1, y=0,
                     steps=steps)]


custom = [[0.0, "rgb(165,0,38)"],
          [0.1111111111111111, "rgb(215,48,39)"],
          [0.2222222222222222, "rgb(244,109,67)"],
          [0.3333333333333333, "rgb(253,174,97)"],
          [0.4444444444444444, "rgb(254,224,144)"],
          [0.5555555555555556, "rgb(224,243,248)"],
          [0.6666666666666666, "rgb(171,217,233)"],
          [0.7777777777777778, "rgb(116,173,209)"],
          [0.8888888888888888, "rgb(69,117,180)"],
          [1.0, "rgb(49,54,149)"]]

class_symbols = np.array(["circle", "x", "diamond"])
class_colors = lambda n: [custom[i] for i in np.linspace(0, len(custom)-1, n).astype(int)]

def decision_surface(predict, xrange, yrange, density=120, dotted=False, colorscale=custom, showscale=True):
    xrange, yrange = np.linspace(*xrange, density), np.linspace(*yrange, density)
    xx, yy = np.meshgrid(xrange, yrange)
    pred = predict(np.c_[xx.ravel(), yy.ravel()])

    if dotted:
        return go.Scatter(x=xx.ravel(), y=yy.ravel(), opacity=1, mode="markers", marker=dict(color=pred, size=1, colorscale=colorscale, reversescale=False), hoverinfo="skip", showlegend=False)
    return go.Contour(x=xrange, y=yrange, z=pred.reshape(xx.shape), colorscale=colorscale, reversescale=False, opacity=.7, connectgaps=True, hoverinfo="skip", showlegend=False, showscale=showscale)


   
def animation_to_gif(fig, filename, frame_duration=100, width=1200, height=800):
    import tempfile

    # Set figure size
    fig.update_layout(width=width, height=height)

    with tempfile.TemporaryDirectory() as tmpdir:
        images = []
        for i, frame in enumerate(fig.frames):
            f = go.Figure(data=frame.data, layout=fig.layout)
            f.update_layout(title=frame.layout.title, width=width, height=height)
            path = os.path.join(tmpdir, f"frame_{i:03d}.png")
            f.write_image(path, format="png")
            images.append(iio.imread(path))

        iio.imwrite(filename, images, duration=frame_duration / 1000.0)  # duration in seconds



def create_data_bagging_utils(d = 4, number_of_members = 1, n_samples = 1000):
    
    def sample_beta(limit1, limit2):
        margin1 = limit1 + (limit2 - limit1)*0.45
        margin2 = limit2 - (limit2 - limit1)*0.45
        beta = np.random.uniform(margin1, margin2)
        return beta

  # Creates n samples
    samples = np.random.uniform(size=(n_samples, 2))

    samples_of_half = "samples_of_half"
    x_1 = "x_1"; x_2 = "x_2"; y_1 = "y_1"; y_2 = "y_2"; tag = "tag"
    list_of_array = {0: {samples_of_half : samples, x_1 : 0, x_2 : 1, y_1 : 0, y_2 : 1}}

    for i in range(0, d):
        built_list =  {}
        for sample_curr_i, sample_curr in enumerate(list_of_array.values()):
      # Choose if we want to split x axis or y axis
            dim_half = np.random.choice([0,1])

            dots_coords = sample_curr[samples_of_half]
            if (dim_half == 0):
                beta = sample_beta(sample_curr[x_1], sample_curr[x_2])
                built_list[sample_curr_i*2] = {samples_of_half: dots_coords[dots_coords[:,0] <= beta],
                                       x_1 : sample_curr[x_1],
                                       x_2 : beta,
                                       y_1 : sample_curr[y_1],
                                       y_2 : sample_curr[y_2],
                                       tag : np.random.choice([0, 1]).astype(int)}

                built_list[sample_curr_i*2 + 1] = {samples_of_half: dots_coords[dots_coords[:,0] > beta],
                                       x_1 : beta,
                                       x_2 : sample_curr[x_2],
                                       y_1 : sample_curr[y_1],
                                       y_2 : sample_curr[y_2],
                                       tag : np.random.choice([0, 1]).astype(int)}
            else:
                beta = sample_beta(sample_curr[y_1], sample_curr[y_2])
                built_list[sample_curr_i*2] = {samples_of_half: dots_coords[dots_coords[:,1] <= beta],
                                       x_1 : sample_curr[x_1],
                                       x_2 : sample_curr[x_2],
                                       y_1 : sample_curr[y_1],
                                       y_2 : beta,
                                       tag : np.random.choice([0, 1]).astype(int)}

                built_list[sample_curr_i*2 + 1] = {samples_of_half: dots_coords[dots_coords[:,1] > beta],
                                       x_1 : sample_curr[x_1],
                                       x_2 : sample_curr[x_2],
                                       y_1 : beta,
                                       y_2 : sample_curr[y_2],
                                       tag : np.random.choice([0, 1]).astype(int)}


        list_of_array =  built_list
    samples = np.vstack([samples_["samples_of_half"] for samples_ in built_list.values()])
    tags =  np.hstack([np.repeat(samples_["tag"], samples_["samples_of_half"].shape[0]) for samples_ in built_list.values()])
    return samples, tags


def plot_decision_boundary(nn: NeuralNetwork, lims, X: np.ndarray = None, y: np.ndarray = None, title=""):
    data = [decision_surface(nn.predict, lims[0], lims[1], density=40, showscale=False)]
    if X is not None:
        col = y if y is not None else "black"
        data += [go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                            marker=dict(color=col, colorscale=custom, line=dict(color="black", width=1)))]

    return go.Figure(data,
                     go.Layout(title=rf"$\text{{Network Decision Boundaries {title}}}$",
                               xaxis=dict(title=r"$x_1$"), yaxis=dict(title=r"$x_2$"),
                               width=400, height=400))


def animate_decision_boundary(nn: NeuralNetwork, weights: List[np.ndarray], lims, X: np.ndarray, y: np.ndarray,
                              title="", save_name=None):
    frames = []
    for i, w in enumerate(weights):
        nn.weights = w
        frames.append(go.Frame(data=[decision_surface(nn.predict, lims[0], lims[1], density=40, showscale=False),
                                     go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                                marker=dict(color=y, colorscale=custom, line=dict(color="black", width=1)))
                                     ],
                               layout=go.Layout(title=rf"$\text{{{title} Iteration {i+1}}}$")))

    fig = go.Figure(data=frames[0]["data"], frames=frames[1:],
                    layout=go.Layout(title=frames[0]["layout"]["title"]))
    if save_name:
        animation_to_gif(fig, save_name, 200, width=400, height=400)