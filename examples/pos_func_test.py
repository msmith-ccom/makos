from matplotlib import pyplot as plt
from makos.sonar.position import position, position_plot
from makos.sonar.shading import shading, plot_1d, plot_2d

if __name__ == '__main__':

    # Create an empty sonar array
    snr_array = dict()

    print(f"snr_array of type {type(snr_array)}")

    # Test the position functions
    snr_array = position(snr_array=snr_array, num_ele_along=72, num_ele_across=6,
                         spacing_across=0.0104, spacing_along=0.0104,
                         lattice='triangular',
                         element_type='tonpilz', radius=.5)

    # fig = position_plot(snr_array, num_dimensions=2)
    # plt.show()

    # Test the array shading functions
    snr_array = shading(snr_array=snr_array, array_dimension='along', window='chebwin',
                        at=25.)
    # snr_array = shading(snr_array=snr_array, array_dimension='along', window='tukey',
    #                     alpha=0.5)
    snr_array = shading(snr_array=snr_array, array_dimension='across', window='chebwin',
                        at=25.)
    # snr_array = shading(snr_array=snr_array, array_dimension='along', window='boxcar')

    fig1 = plot_1d(snr_array, 1)

    fig2 = plot_2d(snr_array)
    plt.show()
    print('Debug point!')