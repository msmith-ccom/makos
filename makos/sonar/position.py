import numpy as np

from matplotlib import pyplot as plt
from typing import Optional
from warnings import warn

dict_lattice_type = {'rectangular': 0, 'triangular': 1}
dict_element_type = {'omni': 0, 'tonpilz': 1, 'lurtonaprx': 2}


def position(snr_array=None, num_ele_along: int = 1, num_ele_across: int = 1,
             spacing_along: float = 0., spacing_across: float = 0.,
             lattice: str = 'rectangular', element_type: str = 'omni',
             radius: Optional[float] = None, delpsi: Optional[float] = None,
             depth: float = 0.) -> dict:
    """ Generates the element positions for sonar array object """

    # Grab the snr_array dictionary and create the postion entry, create if not passed
    if snr_array is None:
        snr_array = {}

    try:
        _ = snr_array['elements']
    except KeyError as e:
        snr_array['elements'] = dict()

    elemnts = snr_array['elements']

    # Determine array dimension sizes
    if num_ele_along < 0:
        raise AttributeError(f"Number of elements along array dimension must be "
                             f"greater than 0: {num_ele_along=}")
    elif num_ele_along == 0:
        num_ele_along = 1

    if (num_ele_along != 1) and (spacing_along <= 0.):
        raise ValueError(f"Invalid element spacing for 'spacing_along'. Must be "
                         f"positve, non-zero float")
    elemnts['num_elements_x'] = num_ele_along
    elemnts['element_spacing_x'] = spacing_along

    if num_ele_across < 0:
        raise AttributeError(f"Number of elements along array dimension must be "
                             f"greater than 0: {num_ele_across=}")
    elif num_ele_across == 0:
        num_ele_across = 1

    if (num_ele_across != 1) and (spacing_across <= 0.):
        raise ValueError(f"Invalid element spacing for 'spacing_across'. Must be "
                         f"positve, non-zero float")

    elemnts['num_elements_y'] = num_ele_across
    elemnts['element_spacing_y'] = spacing_across

    # Check rest of inputs
    if lattice not in dict_lattice_type.keys():
        raise ValueError(f"Invalid input for lattice. Must be either: "
                         f"{dict_lattice_type.keys()}")
    else:
        if (num_ele_along == 1) or (num_ele_across == 1):
            # One of the array dimensions is singleton. Force rectangular array
            lattice = 'rectangular'
            warn(f"One Array dimension is singular, triangular lattice is not "
                 f"possible. Setting lattice to rectangular")

    elemnts['lattice_type'] = lattice
    lat_id = dict_lattice_type[lattice]

    # Define array element type
    if element_type not in dict_element_type.keys():
        raise ValueError(f"Invalid input for lattice, Must be : "
                         f"{dict_element_type.keys()}")

    ele_id = dict_element_type[element_type]

    if ele_id == dict_element_type['tonpilz']:
        if radius is None:
            raise RuntimeError(f"Element type tonpilz requires input: radius")
        elif radius <= 0:
            raise ValueError(f"Input radius must be positive, non-zero number")
    elif ele_id == dict_element_type['lurtonaprx']:
        if delpsi is None:
            raise RuntimeError(f"Element type lurtonaprox requires input: delpsi")
        elif delpsi <= 0:
            raise ValueError(f"Input delpsi must be positive, non-zero number")

    # Fill output element structure (I should just make this all classes)
    elemnts['element'] = dict()
    ele = elemnts['element']
    ele['element_type'] = element_type
    ele['delpsi'] = delpsi
    ele['radius'] = radius

    # Set array depth
    # *** Note on axis convention:
    #   x = to bow/ forward + / ship relative
    #   y = to starboard/ right + / ship relative

    if depth > 0:
        depth = abs(depth)
    elemnts['depth'] = depth

    # Establish element positions - rectangular lattice
    if lat_id == 0:
        i_x = np.arange(0, num_ele_along)[:, np.newaxis]
        x = i_x * spacing_along
        mid = x.max() / 2
        x = x - mid

        i_y = np.arange(0, num_ele_across)[np.newaxis, :]
        y = i_y * spacing_across
        mid = y.max() / 2
        y = y - mid

        # Let's store them
        y_out, x_out = np.meshgrid(y, x)
        z_out = depth * np.ones(shape=y_out.shape)

    else:
        i_x = np.arange(0, 2*num_ele_along)[:, np.newaxis]
        x = i_x * (spacing_along / 2)

        i_y = np.arange(0, 2*num_ele_across)[np.newaxis, :]
        y = i_y * (spacing_across / 2)

        [yy, xx] = np.meshgrid(y, x)
        zz = depth * np.ones(shape=xx.shape)

        x_out = np.nan * np.ones(shape=xx.shape)
        y_out = x_out.copy()
        z_out = x_out.copy()

        x_out[::2, ::2] = xx[::2, ::2]
        x_out[1::2, 1::2] = xx[1::2, 1::2]
        y_out[::2, ::2] = yy[::2, ::2]
        y_out[1::2, 1::2] = yy[1::2, 1::2]
        z_out[::2, ::2] = zz[::2, ::2]
        z_out[1::2, 1::2] = zz[1::2, 1::2]

        x_out = x_out - (x.max() / 2)
        y_out = y_out - (y.max() / 2)

    out_dims = list(x_out.shape)
    out_dims.append(3)
    pos = np.zeros(shape=out_dims)

    pos[:, :, 0] = x_out
    pos[:, :, 1] = y_out
    pos[:, :, 2] = z_out

    elemnts['position'] = pos

    return snr_array


def position_plot(snr_array, num_dimensions: int = 2):
    """ Plot the element positions """

    if num_dimensions not in [2, 3]:
        raise ValueError(f"num_dimensions must be either 2 or 3.")

    # Grab the element positions
    try:
        pos = snr_array['elements']['position']
    except KeyError as e:
        raise KeyError(f"Make sure position function has been run") from e

    # Create the figure
    fig = plt.figure()

    # Plot the figure
    if num_dimensions == 2:
        ax = fig.add_subplot()
        ax.scatter(pos[:, :, 1], pos[:, :, 0])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')
        lim_left, lim_right = plt.xlim()
        plt.xlim((lim_left-0.25, lim_right+0.25))

    else:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(pos[:, :, 0], pos[:, :, 1], pos[:, :, 2])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

        limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
        ax.set_box_aspect(np.ptp(limits + 0.25, axis=1))

    ax.set_title('Array Element Positions')

    return fig
