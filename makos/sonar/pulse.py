import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from scipy.signal import get_window, hilbert, freqz
from scipy.fft import next_fast_len
from typing import Optional


def set_pulse(snr_array: dict, fc: float, amplitude: float = 1,
              tau: float = 0.010, fs: float = 120000,
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
    else:  # There are extra parameters, packaged in tuple
        win_params = (shading,) + tuple(kwargs.values())
        pulse_window = get_window(win_params, num_samples, fftbins=False)

    pulse_setting['pulse_length'] = tau
    pulse_setting['center_frequency'] = fc
    pulse_setting['sample_rate'] = fs
    pulse_setting['amplitude'] = amplitude
    pulse_setting['pulse_taper'] = shading
    pulse_setting['pulse_taper_terms'] = pulse_window

    return snr_array


def gen_pulse(snr_array: dict, sound_speed: float = 1500, distance: float = 0.,
              analytic: bool = False):
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

    val_time = np.arange(0, tau, dt)

    # Check for matching lengths -> issue can come about due to needing whole integer
    # for window lengths
    offset = len(val_time) - len(wn)
    if offset > 0:
        val_time = val_time[:-offset]
    elif offset < 0:
        for ii in range(offset, ):
            val_time = np.append(val_time, [val_time[-1] + dt])
    else:
        pass

    waveform = a * np.cos((w * val_time) - (k * distance))

    if analytic:
        waveform = hilbert(waveform)

    # Modulate using the window
    waveform = wn * waveform
    return val_time, waveform, fs


def plot_pulse(snr_array: Optional[dict] = None,
               sample: Optional[np.ndarray] = None,
               amplitude: Optional[np.ndarray] = None,
               sound_speed: float = 1500, distance: float = 0.):
    """ Plots the waveform"""

    if snr_array is not None:
        sample, amplitude, _ = gen_pulse(snr_array, sound_speed=sound_speed,
                                         distance=distance, analytic=True)
    else:
        if (sample is None) or (amplitude is None):
            raise ValueError(f"sample and amplitude, or snr_array is None. Cannot "
                             f"generate pulse")

    # Generate the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        name="Real Signal",
        x=sample, y=np.real(amplitude),
        mode='lines'
    ))
    fig.add_trace(go.Scatter(
        name="Signal Envelope",
        x=sample, y=np.abs(amplitude),
        mode='lines',
    ))
    fig.update_layout(
        title=dict(
            text='Array Waveform',
            x=0.5
        ),
        xaxis_title="Time in seconds", yaxis_title="Amplitude")

    return fig


def plot_spectrum(snr_array: Optional[dict] = None,
                  sample: Optional[np.ndarray] = None,
                  amplitude: Optional[np.ndarray] = None,
                  sound_speed: float = 1500, distance: float = 0., fs: int = 120000):
    if snr_array is not None:
        sample, amplitude, fs = gen_pulse(snr_array, sound_speed=sound_speed,
                                          distance=distance, analytic=True)
    else:
        if (sample is None) or (amplitude is None):
            raise ValueError(f"sample and amplitude, or snr_array is None. Cannot "
                             f"generate pulse")

    # Calculate the spectrum
    frqs, h_response = freqz(b=amplitude, a=1, worN=next_fast_len(len(amplitude)),
                             whole=False, fs=fs)

    angles = np.unwrap(np.angle(h_response))

    #
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=frqs / 1000, y=20 * np.log10(np.abs(h_response)),
        name="Frequency response",
        mode='lines'
        ),
        secondary_y=False
    )

    fig.add_trace(go.Scatter(
        x=frqs / 1000, y=angles,
        name="Phase Angle",
        mode='lines'
        ),
        secondary_y=True
    )

    fig.update_layout(title_text="Spectral analysis of transmit signal",
                      xaxis_title="Frequency (kHz)")
    fig.update_yaxes(title_text="Response Magnitude (dB)", secondary_y=False)
    fig.update_yaxes(title_text="Phase Angle (radians)", secondary_y=True)


    return fig


def plot_spectrogram():
    raise NotImplementedError(f"This function is not yet implemented")

# ## Private methods ##
