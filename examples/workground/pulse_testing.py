from makos.sonar.position import position
from makos.sonar.pulse import set_pulse, plot_pulse, plot_spectrum, plot_spectrogram


if __name__ == '__main__':

    # Create an empty sonar array
    snr_array = dict()

    print(f"snr_array of type {type(snr_array)}")

    # ## Test the position functions ##
    snr_array = position(snr_array=snr_array, num_ele_along=72, num_ele_across=6,
                         spacing_across=0.104, spacing_along=0.104,
                         lattice='triangular',
                         element_type='lurtonaprx', delpsi=107)

    # ## Test the pulse functions ##
    snr_array = set_pulse(snr_array, amplitude=1.0, tau=0.020, fc=12000, fs=220000,
                          shading='tukey', alpha=0.25)

    fig = plot_pulse(snr_array)
    fig.show()
    fig = plot_spectrum(snr_array)
    fig.show()