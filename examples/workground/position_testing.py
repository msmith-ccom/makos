from makos.sonar.position import position, pos_plot_2d, pos_plot_3d

if __name__ == '__main__':

    # Create an empty sonar array
    snr_array = dict()

    print(f"snr_array of type {type(snr_array)}")

    # ## Test the position functions ##
    snr_array = position(snr_array=snr_array, num_ele_along=72, num_ele_across=6,
                         spacing_across=0.104, spacing_along=0.104,
                         lattice='triangular',
                         element_type='lurtonaprx', delpsi=107)

    fig = pos_plot_2d(snr_array)
    fig2 = pos_plot_3d(snr_array)
    fig.show()
    fig2.show()

