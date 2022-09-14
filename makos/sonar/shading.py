import numpy as np

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

    if array_dimension in (0, 'x', 'along'):
        dim = 0
    elif array_dimension in (1, 'y', 'across'):
        dim = 1
    else:
        raise ValueError(f"Invalid input 'array_dimension'. Must be either ('x', "
                         f"'along', 0) or ('y', 'across', 1)")

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
