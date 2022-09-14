import numpy as np

from matplotlib import pyplot as plt
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

    else:       # There are extra parameters, wrap in tuple
        win_params = (window,) + tuple(kwargs.values())
        window_terms = get_window(win_params, dims[dim], fftbins=False)

    # ## Fill output dictionary and combine window terms if necessary ##
    if dim == 0:     # Along shading
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

    elif dim == 1:     # Across shading
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

    fig = plt.figure()
    ax = fig.add_subplot()

    if dim == 0:
        try:
            ax.stem(np.arange(dims[dim]), snr_array['shading']['along_window_terms'],
                    basefmt=' ')
        except KeyError as e:
            raise KeyError(f"Array along shading terms not found. Make sure funct: "
                           f"shading has been run") from e
    else:
        try:
            ax.stem(np.arange(dims[dim]), snr_array['shading']['across_window_terms'],
                    basefmt=' ')
        except KeyError as e:
            raise KeyError(f"Array across shading terms not found. Make sure funct: "
                           f"shading has been run") from e

    if snr_array['elements']['lattice_type'] == 'rectangular':
        title_str = f"{dim_title[dim]} Shading Terms"
    else:
        title_str = f"{dim_title[dim]} Shading Terms \n Note element number double " \
                    f"due to triangular lattice"
    ax.set_title(title_str)
    ax.set_xlabel("Element #")
    ax.set_ylabel("Element shading coefficient")
    ax.set_ylim(bottom=0.0)
    ax.grid(visible=True)
    return fig


def plot_2d(snr_array: dict):
    """ Plots the 2D Array shading terms as color image"""

    # Get out the data
    shade_weights = snr_array['shading']['array_weights']
    dims = shade_weights.shape

    fig, (ax, cax) = plt.subplots(ncols=2, nrows=1,
                                  figsize=(2, 4),
                                  gridspec_kw={'width_ratios': [1, 0.05]})
    img = ax.imshow(shade_weights)
    cbar = fig.colorbar(img, cax=cax, orientation='vertical')
    cbar.set_label("Shading coefficient")

    if snr_array['elements']['lattice_type'] == 'rectangular':
        title_str = f"Shading Terms"
    else:
        title_str = f"Shading Terms \n Note element number double " \
                    f"due to triangular lattice"
    ax.set_title(title_str)
    ax.set_xlabel("Across element #")
    ax.set_ylabel("Along element #")

    x_minor = np.arange(dims[1])
    y_minor = np.arange((dims[0]))
    ax.set_xticks(x_minor, minor=True)
    ax.set_yticks(y_minor, minor=True)
    ax.grid(visible=True, which='both', color='white', alpha=0.5)

    ax.set_aspect('equal')
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
