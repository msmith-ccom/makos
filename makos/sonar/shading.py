import numpy as np
import plotly.graph_objects as go

from scipy.signal import get_window
from typing import Union


def shading(snr_array: dict, array_dimension: Union[str, int],
            window: str, **kwargs) -> dict:
    """ Determines the shading elements for the input array

    This function calculates the shading terms for the input array. Function defines
    terms for a single array dimension, but will update/combine shading terms when
    shading multiple dimensions.

    The function is a lite wrapper over the scipy.signal.windows toolbox. All windows
    listed in the toolbox can be used. For windows which have extra parameters,
    pass them as keyword arguments. See:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.get_window.html#scipy.signal.windows.get_window
    for more information on the window

    """

    # ## Check for valid inputs ##
    if 'position' not in snr_array['elements'].keys():
        raise KeyError(f"Array element position data not found. Run position() ")

    dim = _check_dim(array_dimension)

    if 'shading' not in snr_array.keys():
        snr_array['shading'] = dict()
    shade_setting = snr_array['shading']

    # Grab out the array dimensions
    dims = snr_array['elements']['position'].shape

    # ## Get the window terms ##
    if not kwargs:
        window_terms = get_window(window, dims[dim], fftbins=False)

    else:  # There are extra parameters, wrap in tuple
        win_params = (window,) + tuple(kwargs.values())
        window_terms = get_window(win_params, dims[dim], fftbins=False)

    # ## Fill output dictionary and combine window terms if necessary ##
    if dim == 0:  # Along shading
        tmp_0 = np.tile(window_terms[:, None], (1, dims[1]))
        if 'array_weights' in shade_setting:
            # We have already run array shading before: Compose new weights.
            try:
                tmp_1 = shade_setting['across_window_terms'][None, :]
                tmp_1 = np.tile(tmp_1, (dims[0], 1))
                shade_setting['array_weights'] = tmp_0 * tmp_1
            except KeyError as e:
                # We are re-runing the shading for this dimension/ across shading not
                # run yet.
                shade_setting['array_weights'] = tmp_0
        else:
            shade_setting['array_weights'] = tmp_0

        shade_setting['along_window_type'] = window
        shade_setting['along_window_terms'] = window_terms

    elif dim == 1:  # Across shading
        tmp_1 = np.tile(window_terms[None, :], (dims[0], 1))
        if 'array_weights' in shade_setting:
            # We have already run array shading before: Compose new weights
            try:
                tmp_0 = shade_setting['along_window_terms'][:, None]
                tmp_0 = np.tile(tmp_0, (1, dims[1]))
                shade_setting['array_weights'] = tmp_0 * tmp_1
            except KeyError as e:
                # We are re-runing the shading for this dimension/ along shading not
                # run yet.
                shade_setting['array_weights'] = tmp_1
        else:
            shade_setting['array_weights'] = tmp_1

        shade_setting['across_window_type'] = window
        shade_setting['across_window_terms'] = window_terms

    else:
        raise NotImplementedError(f"There is no support for dimensions beyond x and y")

    return snr_array


def plot_1d(snr_array: dict, array_dim: Union[str, int]):
    """ Plots the element shading terms along a given direction """

    dim_title = {
        0: "Along",
        1: "Across",
    }

    dim = _check_dim(array_dim)
    dims = snr_array['elements']['position'].shape

    if dim == 0:
        try:
            window_terms = snr_array['shading']['along_window_terms']
        except KeyError as e:
            raise KeyError(f"Array along shading terms not found. Make sure funct: "
                           f"shading has been run") from e
    else:
        try:
            window_terms = snr_array['shading']['across_window_terms']
        except KeyError as e:
            raise KeyError(f"Array across shading terms not found. Make sure funct: "
                           f"shading has been run") from e

    if snr_array['elements']['lattice_type'] == 'rectangular':
        title_str = f"{dim_title[dim]} Shading Terms"
    else:
        title_str = f"{dim_title[dim]} Shading Terms <br> " \
                    f"Note element number double due to triangular lattice"
    window_index = np.arange(dims[dim])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=window_index, y=window_terms,
        mode='markers',
        marker=dict(
            size=10,
            line_width=2,
            color='black'
        ),
        error_y=dict(
            type='data',
            symmetric=False,
            array=np.zeros(shape=window_index.shape),
            arrayminus=window_terms,
            width=0
        ),
        hovertemplate="(%{x:d},%{y:.5f})<extra></extra>"
    ))

    fig.update_layout(xaxis_title="Window term (n)",
                      yaxis_title="Window term amplitude",
                      yaxis_rangemode='tozero',
                      title=dict(
                          text=title_str,
                          x=0.5
                      ),
                      showlegend=False
                      )
    return fig


def plot_2d(snr_array: dict):
    """ Plots the 2D Array shading terms as color image"""

    # Get out the data
    shade_weights = snr_array['shading']['array_weights']
    x = snr_array['elements']['position'][:, :, 0]
    dims = list(shade_weights.shape)

    # Mask data
    nan_inds = np.isnan(x)
    shade_weights[nan_inds] = np.nan

    if dims[0] > dims[1]:
        shade_weights = shade_weights.T
        dims = list(reversed(dims))

    if snr_array['elements']['lattice_type'] == 'rectangular':
        title_str = f"Shading Terms"
    else:
        title_str = f"Shading Terms <br>" \
                    f" Note element number double due to triangular lattice"

    # Plot the figure
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=shade_weights, x=list(range(dims[0])), y=list(range(dims[1])),
        colorscale='Viridis',
        colorbar=dict(
            title="Shading coefficient"
        ),
        xgap=1, ygap=1
    ))
    fig.update_layout(
        xaxis=dict(title="Array long axis index (n)"),
        yaxis=dict(
            scaleanchor='x',
            title="Array shor axis index (m)"
        ),
        title=dict(
            text=title_str,
            x=0.5
        )

    )

    return fig


def _check_dim(array_dimension) -> int:
    """ Internal function to check dimension input"""
    if array_dimension in (0, 'x', 'along'):
        dim = 0
    elif array_dimension in (1, 'y', 'across'):
        dim = 1
    else:
        raise ValueError(f"Invalid input 'array_dimension'. Must be either ('x', "
                         f"'along', 0) or ('y', 'across', 1)")
    return dim
