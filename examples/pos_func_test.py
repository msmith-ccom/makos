from matplotlib import pyplot as plt
from makos.sonar.position import position, position_plot


if __name__ == '__main__':

    # Create an empty sonar array
    snr_array = dict()

    print(f"snr_array of type {type(snr_array)}")

    snr_array = position(snr_array=snr_array, num_ele_along=72, num_ele_across=6,
                         spacing_across=0.0104, spacing_along=0.0104,
                         lattice='triangular',
                         element_type='tonpilz', radius=.5)

    fig = position_plot(snr_array, num_dimensions=5)
    plt.show()
