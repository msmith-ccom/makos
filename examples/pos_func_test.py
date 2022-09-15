from matplotlib import pyplot as plt
from makos.sonar.position import position, position_plot
from makos.sonar.shading import shading, plot_1d, plot_2d
from makos.sonar.pulse import pulse, plot_pulse

if __name__ == '__main__':

    # Create an empty sonar array
    snr_array = dict()

    print(f"snr_array of type {type(snr_array)}")

    # ## Test the position functions ##
    snr_array = position(snr_array=snr_array, num_ele_along=72, num_ele_across=6,
                         spacing_across=0.0104, spacing_along=0.0104,
                         lattice='triangular',
                         element_type='tonpilz', radius=.5)

    # fig = position_plot(snr_array, num_dimensions=2)
    # plt.show()

    # ## Test the array shading functions ##
    snr_array = shading(snr_array=snr_array, array_dimension='along', window='chebwin',
                        at=25.)
    # snr_array = shading(snr_array=snr_array, array_dimension='along', window='tukey',
    #                     alpha=0.5)
    snr_array = shading(snr_array=snr_array, array_dimension='across', window='chebwin',
                        at=25.)
    # snr_array = shading(snr_array=snr_array, array_dimension='along', window='boxcar')

    # fig1 = plot_1d(snr_array, 1)
    #
    # fig2 = plot_2d(snr_array)
    # plt.show()

    # ## Test the pulse functions ##
    snr_array = pulse(snr_array, amplitude=1.0, tau=0.020, fc=12000, fs=120000,
                      shading='tukey', alpha=0.25)

    fig = plot_pulse(snr_array, sound_speed=1500, analytic=True)
    plt.show()
    print('Debug point!')