import numpy as np

from matplotlib import pyplot as plt
from scipy.signal import get_window, hilbert


def pulse(snr_array: dict, amplitude: float, tau: float, fc: float, fs: float,
          shading: str = 'boxcar', **kwargs) -> dict:

    if 'pulse' not in snr_array.keys():
        snr_array['pulse'] = dict()
    pulse_setting = snr_array['pulse']

    num_samples = tau * fs
    if not num_samples.is_integer():
        num_samples = int(num_samples) + 1
    else:
        num_samples = int(num_samples)

    # Get the shading window
    if not kwargs:
        pulse_window = get_window(shading, num_samples, fftbins=False)
    else:       # There are extra parameters, package in tuple
        win_params = (shading,) + tuple(kwargs.values())
        pulse_window = get_window(win_params, num_samples, fftbins=False)

    pulse_setting['pulse_length'] = tau
    pulse_setting['center_frequency'] = fc
    pulse_setting['sample_rate'] = fs
    pulse_setting['amplitude'] = amplitude
    pulse_setting['pulse_taper'] = shading
    pulse_setting['pulse_taper_terms'] = pulse_window

    return snr_array


def plot_pulse(snr_array, sound_speed: float = 1500, range: float = 0.,
               analytic: bool = False):
    # Unpack dictionary
    try:
        pulse_setting = snr_array['pulse']
        a = pulse_setting['amplitude']
        fs = pulse_setting['sample_rate']
        fc = pulse_setting['center_frequency']
        tau = pulse_setting['pulse_length']
        wn = pulse_setting['pulse_taper_terms']
    except KeyError as e:
        raise KeyError(f"Could not unpack snr_array, make sure to run pulse()") from e

    # Compute the wave form - start with a real valued sign wave
    dt = 1 / fs
    w = 2 * np.pi * fc
    k = w / sound_speed

    time = np.arange(0, tau, dt)

    # Check for matching lengths -> issue can come about due to needing whole integer
    # for window lengths
    offset = len(time) - len(wn)
    if offset > 0:
        time = time[:-offset]
    elif offset < 0:
        for ii in range(offset,):
            time = np.append(time, [time[-1]+dt])
    else:
        pass

    waveform = a * np.cos((w * time) - (k * range))

    if analytic:
        waveform = hilbert(waveform)

    # Modulate using the window
    waveform = wn * waveform

    # Generate the figure
    fig: plt.Figure = plt.figure()

    if analytic:
        ax0: plt.Axes = fig.add_subplot(211)
        ax0.plot(time, np.real(waveform), label='real signal')
        ax0.plot(time, np.abs(waveform), label='envelope')
        ax0.legend()
        ax0.set_xlabel(f"time in seconds")
        ax0.set_ylabel(f"amplitude")
        ax0.set_title(f"Real waveform and envelope")

        ax1: plt.Axes = fig.add_subplot(212)
        instant_frq = np.diff(np.unwrap(np.angle(waveform))) / (2.0 * np.pi) * fs
        ax1.plot(time[1:], instant_frq)
        ax1.set_ylim(bottom=instant_frq.min()-20, top=instant_frq.max()+20)
        ax1.set_xlabel(f"time in seconds")
        ax1.set_ylabel(f"frequency in Hz")
        ax1.set_title(f"Waveform frequency vs time")
        fig.suptitle(f"Signal Visualization")

    else:
        ax0: plt.Axes = fig.add_subplot()
        ax0.plot(time, waveform)
        ax0.set_xlabel(f"time in seconds")
        ax0.set_ylabel(f"amplitude")
        ax0.set_title(f"Waveform")

    return fig
