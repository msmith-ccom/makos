import numpy as np
import plotly.graph_objects as go

from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.special import jv
from typing import Union, Optional
from warnings import warn

from makos.sonar.position import dict_element_type

dict_bearing_str = {
    'along': 0.,
    'across': 90.
}

dict_slice_plt = {
    'linear': 0,
    'polar': 1
}


# ### Radiation pattern calculation methods ###
def calc_slice(snr_array, c_range: float = 1500., sound_speed: float = 1500.,
               center_frequency: Optional[float] = None,
               slice_bearing: Union[str, float] = 0., cut_angle: float = 0.,
               focus: float = 0., source_level: Optional[float] = None,
               along_steering: float = 0., across_steering: float = 0.) -> tuple:
    # Parse inputs
    bearing = _check_bearing(slice_bearing=slice_bearing)

    try:
        x_array = snr_array['elements']['position'][:, :, 0]
        y_array = snr_array['elements']['position'][:, :, 1]
        z_array = snr_array['elements']['position'][:, :, 2]
    except KeyError as e:
        raise KeyError(f"Could not find array element positions, run position()") from e

    # Define initial x,y,z points at bearing 0.0 for the calculation
    theta_fp = np.round(np.arange(-90, 90.1, 0.1), 2)
    x = c_range * np.sin(np.radians(theta_fp)) * np.cos(np.radians(0.0))
    y = c_range * np.sin(np.radians(theta_fp)) * np.sin(np.radians(0.0))
    z = c_range * np.cos(np.radians(theta_fp))
    field_points = np.array([x, y, z]).T

    if (bearing != 0.0) or (cut_angle != 0.0):
        c_rot = Rotation.from_euler('ZX', [bearing, cut_angle], degrees=True)
        field_points = c_rot.apply(field_points, )

    # Calculate the steering time delays
    delay_along = (x_array * np.sin(np.radians(along_steering))) / sound_speed
    delay_across = (y_array * np.sin(np.radians(across_steering))) / sound_speed

    # Calculate the focus delays
    # NOTE: It may seem strange to have the max focus delay at the central part of
    # the array but recall that we want the elemental transmissions to arrive at the
    # same time, so we delay the center so the edges have the chance to propagate to
    # the extra distance.
    # TODO: Check that focusing along the vertical is okay and that focusing effect
    #  stays along the main beam direction after applying steering delays.
    if focus != 0.:
        zyx_focus = np.array([x_array, y_array, z_array - focus])
        range_diff = np.sqrt(np.sum(zyx_focus ** 2, axis=2))
        delay_focus = -(range_diff - np.nanmax(range_diff)) / sound_speed
    else:
        delay_focus = np.zeros(shape=delay_across.shape)

    delay_total = delay_focus + delay_along + delay_across

    # Set the amplitude and frequency
    if source_level:
        a = 1.
    else:
        try:
            a = snr_array['pulse']['amplitude']
        except KeyError:
            a = 1.
    if center_frequency:
        fc = center_frequency
    else:
        try:
            fc = snr_array['pulse']['center_frequency']
        except KeyError as e:
            raise KeyError(f"Array Frequency not set, either pass center frequency n "
                           f"call or through the makos.pulse function") from e

    # Calculate the unshaded waveform for each element
    x = field_points[:, 0]
    x = np.tile(x[None, :], (len(x_array.flatten()), 1))
    y = field_points[:, 1]
    y = np.tile(y[None, :], (len(y_array.flatten()), 1))
    z = field_points[:, 2]
    z = np.tile(z[None, :], (len(z_array.flatten()), 1))

    range_prime = np.sqrt((x - x_array.flatten()[:, None]) ** 2 +
                          (y - y_array.flatten()[:, None]) ** 2 +
                          (z - z_array.flatten()[:, None]) ** 2)

    w = 2 * np.pi * fc
    k = w / sound_speed
    exp_wvfm = np.exp(1j * (w * -delay_total.flatten()[:, None] - k * range_prime))
    exp_wvfm = (a / range_prime) * exp_wvfm

    # Shade the array
    try:
        wn = snr_array['shading']['array_weights']
        exp_wvfm = wn.flatten()[:, None] * exp_wvfm
    except KeyError:
        warn(f"No shading parameters found, array assumed unshaded")

    # ## Calculate the Element beam pattern ##
    try:
        ele_flag = dict_element_type[snr_array['elements']['element']['element_type']]
    except KeyError:
        warn(f"Could not find element type, assume omnidirectional element")
        ele_flag = 0

    # Determine angle between element normal and field point
    psi = np.degrees(np.arccos((z - z_array.flatten()[:, None]) / range_prime))

    inx_0 = np.argwhere(psi == 0)
    psi[inx_0] = np.finfo(float).eps

    if ele_flag == 0:
        ele_patt = np.ones(shape=exp_wvfm.shape)
    elif ele_flag == 1:
        radius = snr_array['elements']['element']['radius']
        mm = k * radius * np.sin(np.radians(psi))
        ele_patt = (2 * jv(1, mm)) / mm
    else:
        del_psi = snr_array['elements']['element']['delpsi']
        e_cap = (np.pi * psi) / del_psi
        ele_patt = np.sin(e_cap) / e_cap

    dir_patt = exp_wvfm * ele_patt
    dir_patt = np.abs(np.nansum(dir_patt, axis=0))

    # ## Normalize output ##
    if source_level is not None:
        dp_max = np.max(dir_patt)
        slice_data = 20. * np.log10(dir_patt / dp_max) + source_level
    else:
        if a == 1.:
            dp_max = np.max(dir_patt)
            slice_data = 20. * np.log10(dir_patt / dp_max)
        else:
            t_loss = 20. * np.log10(c_range)
            slice_data = 20. * np.log10(dir_patt) + t_loss

    return theta_fp, slice_data


def calc_3d_pattern(snr_array, c_range: float = 1500., sound_speed: float = 1500.,
               center_frequency: Optional[float] = None,
               focus: float = 0., source_level: Optional[float] = None,
               along_steering: float = 0., across_steering: float = 0.) -> tuple:

    try:
        x_array = snr_array['elements']['position'][:, :, 0]
        y_array = snr_array['elements']['position'][:, :, 1]
        z_array = snr_array['elements']['position'][:, :, 2]
    except KeyError as e:
        raise KeyError(f"Could not find array element positions, run position()") from e

    # Calculate the steering time delays
    delay_along = (x_array * np.sin(np.radians(along_steering))) / sound_speed
    delay_across = (y_array * np.sin(np.radians(across_steering))) / sound_speed

    # Calculate the focus delays
    # NOTE: It may seem strange to have the max focus delay at the central part of
    # the array but recall that we want the elemental transmissions to arrive at the
    # same time, so we delay the center so the edges have the chance to propagate to
    # the extra distance.
    # TODO: Check that focusing along the vertical is okay and that focusing effect
    #  stays along the main beam direction after applying steering delays.
    if focus != 0.:
        zyx_focus = np.array([x_array, y_array, z_array - focus])
        range_diff = np.sqrt(np.sum(zyx_focus ** 2, axis=2))
        delay_focus = -(range_diff - np.nanmax(range_diff)) / sound_speed
    else:
        delay_focus = np.zeros(shape=delay_across.shape)

    delay_total = delay_focus + delay_along + delay_across

    try:
        wn = snr_array['shading']['array_weights']
    except KeyError:
        warn(f"No shading parameters found, array assumed unshaded")
        wn = np.ones(shape=x_array.shape)

    # Set the amplitude and frequency
    if source_level:
        a = 1.
    else:
        try:
            a = snr_array['pulse']['amplitude']
        except KeyError:
            a = 1.
    if center_frequency:
        fc = center_frequency
    else:
        try:
            fc = snr_array['pulse']['center_frequency']
        except KeyError as e:
            raise KeyError(
                f"Array Frequency not set, either pass center frequency n "
                f"call or through the makos.pulse function") from e

    # Define initial angles for calculation
    # TODO: Change back to standard resolution before commit
    theta = np.round(np.arange(0., 90.1, 0.1), 2)
    phi = np.round(np.arange(0, 360.25, 0.25), 2)
    num_theta = len(theta)
    num_phi = len(phi)
    theta[0] = np.finfo(float).eps
    phi[0] = np.finfo(float).eps
    [phi_mesh, theta_mesh] = np.meshgrid(phi, theta)

    w = 2 * np.pi * fc
    k = w / sound_speed

    # Calculate the element beam pattern, cylindrical symmetry is assumed
    try:
        ele_flag = dict_element_type[snr_array['elements']['element']['element_type']]
    except KeyError:
        warn(f"Could not find element type, assume omnidirectional element")
        ele_flag = 0

    if ele_flag == 0:
        ele_patt = np.ones(shape=theta.shape)
    elif ele_flag == 1:
        radius = snr_array['elements']['element']['radius']
        mm = k * radius * np.sin(np.radians(theta))
        ele_patt = np.abs((2 * jv(1, mm)) / mm)
    else:
        del_psi = snr_array['elements']['element']['delpsi']
        e_cap = (np.pi * theta) / del_psi
        ele_patt = np.abs(np.sin(e_cap) / e_cap)

    # Calculate the field points for the calculation
    x = c_range * np.sin(np.radians(theta_mesh)) * np.cos(np.radians(phi_mesh))
    y = c_range * np.sin(np.radians(theta_mesh)) * np.sin(np.radians(phi_mesh))
    z = c_range * np.cos(np.radians(theta_mesh))

    # Reshape the array position data
    x_array = np.tile(x_array.flatten()[:, None], (1, num_theta))
    y_array = np.tile(y_array.flatten()[:, None], (1, num_theta))
    z_array = np.tile(z_array.flatten()[:, None], (1, num_theta))
    wn_array = np.tile(wn.flatten()[:, None], (1, num_theta))

    dim_0, dim_1 = x_array.shape
    df_combined = np.nan * np.ones(shape=(num_theta, num_phi))
    for ii in range(num_phi):

        x_field = np.tile(x[:, ii][None, :], (dim_0, 1))
        y_field = np.tile(y[:, ii][None, :], (dim_0, 1))
        z_field = np.tile(z[:, ii][None, :], (dim_0, 1))

        # Calculate the unshaded element wave form value at range
        range_prime = np.sqrt((x_field-x_array)**2 +
                              (y_field-y_array)**2 +
                              (z_field-z_array)**2)
        exp_wvfm = np.exp(1j * (w*-delay_total.flatten()[:, None] - k*range_prime))
        exp_wvfm = (a / range_prime) * exp_wvfm
        exp_wvfm = wn_array * exp_wvfm
        df_smp_src = np.abs(np.nansum(exp_wvfm, axis=0))
        df_combined[:, ii] = df_smp_src * ele_patt

    # ## Normalize output ##
    if source_level is not None:
        dp_max = np.nanmax(df_combined)
        pattern_data = 20. * np.log10(df_combined / dp_max) + source_level
    else:
        if a == 1.:
            dp_max = np.max(df_combined)
            pattern_data = 20. * np.log10(df_combined / dp_max)
        else:
            t_loss = 20. * np.log10(c_range)
            pattern_data = 20. * np.log10(df_combined) + t_loss

    return theta, phi, pattern_data


# ### Radiation pattern plotting methods ###
def plot_slice(snr_array: dict, plt_type: str = 'linear',
               c_range: float = 1500., sound_speed: float = 1500.,
               center_frequency: Optional[float] = None,
               slice_bearing: Union[str, float] = 0., cut_angle: float = 0.,
               focus: float = 0., source_level: Optional[float] = None,
               along_steering: float = 0., across_steering: float = 0.):
    if plt_type not in dict_slice_plt:
        raise KeyError(f"Unrecognized input 'plt_type: {plt_type}. Must be either: "
                       f"{dict_slice_plt.keys()}")
    else:
        plt_flag = dict_slice_plt[plt_type]
    bearing = _check_bearing(slice_bearing=slice_bearing)

    theta, data_slice = calc_slice(snr_array, c_range=c_range, sound_speed=sound_speed,
                                   center_frequency=center_frequency,
                                   slice_bearing=slice_bearing, cut_angle=cut_angle,
                                   focus=focus, source_level=source_level,
                                   along_steering=along_steering,
                                   across_steering=across_steering)

    if plt_flag == 0:
        fig = plt.figure()
        plt.subplot(111)
        plt.plot(theta, data_slice, linewidth=2)
        plt.xlim((-90, 90))
        plt.xlabel(f"Angle (degrees)")
        plt.ylabel(f"Magnitude (dB re Arbitrary)")
        plt.title(f"Array Beam Pattern Slice \n"
                  f"{bearing=} | {cut_angle=}")
    else:
        # Set the polar plot bounds
        lim_max = data_slice.max()
        lim_min = lim_max - 60

        fig = plt.figure()
        ax: plt.PolarAxes = plt.subplot(111, projection='polar')
        ax.set_rlim(bottom=lim_min, top=lim_max)
        ax.set_theta_zero_location('N')
        ax.plot(np.radians(theta), data_slice)
        ax.set_rlabel_position(-135)
        ax.set_title(f"Array Beam Pattern Slice \n"
                     f"{bearing=} | {cut_angle=}")

    return fig


def plot_3d():
    pass


# ### Private Functions ###

def _check_bearing(slice_bearing: Union[str, float]):
    # Parse inputs
    if type(slice_bearing) is str:
        if slice_bearing not in dict_bearing_str:
            raise ValueError(f"String input for 'bearing must be: "
                             f"{dict_bearing_str.keys()}")
        else:
            bearing = dict_bearing_str[slice_bearing]
    else:
        bearing = slice_bearing
    return bearing


def _sph2cart(az: np.ndarray, ele: np.ndarray, r: np.ndarray) -> tuple:
    """ Converts spherical coordinates to cartesian x,y,z coordinates"""

    try:
        # Data was passed in and can be directly broadcast
        x = r * np.sin(np.radians(ele)) * np.cos(np.radians(az))
        y = r * np.sin(np.radians(ele)) * np.sin(np.radians(az))
        z = r * np.cos(np.radians(ele))
    except ValueError as e:
        # Data cannot be broadcast as is, check for matching dimensions
        r_shape = r.shape
        if len(r_shape) > 2:
            raise NotImplementedError(f"Function does not currently support 3D+ "
                                      f"matricies")
        else:
            az_shape = az.shape
            if az_shape[0] == r_shape[0]:
                az = az[:, None]
                ele = ele[None, :]
            else:
                az = az[None, :]
                ele = ele[:, None]
            x = r * np.sin(np.radians(ele)) * np.cos(np.radians(az))
            y = r * np.sin(np.radians(ele)) * np.sin(np.radians(az))
            z = r * np.cos(np.radians(ele))

    return x, y, z
