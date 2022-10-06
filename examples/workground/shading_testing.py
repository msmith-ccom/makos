from makos.sonar.position import position
from makos.sonar.shading import shading, plot_1d, plot_2d

if __name__ == '__main__':

    # ## Setup the array positions ##
    snr_array = position(num_ele_along=72, num_ele_across=6,
                         spacing_across=0.104, spacing_along=0.104,
                         lattice='triangular',
                         element_type='lurtonaprx', delpsi=107)
# ## Test the array shading functions ##
    snr_array = shading(snr_array=snr_array, array_dimension='along', window='chebwin',
                        at=25.)
    fig1 = plot_1d(snr_array, array_dim=0)
    fig1.show()

    try:
        fig2 = plot_1d(snr_array, array_dim=1)
        fig2.show()
    except KeyError as e:
        print("testing for undefined array dimension passed")

    snr_array = shading(snr_array=snr_array, array_dimension='along', window='tukey',
                        alpha=0.5)
    fig3 = plot_1d(snr_array, array_dim=0)
    fig3.show()
    snr_array = shading(snr_array=snr_array, array_dimension='across', window='chebwin',
                        at=25.)
    fig2 = plot_1d(snr_array, array_dim=1)
    fig2.show()
    snr_array = shading(snr_array=snr_array, array_dimension='along',
                        window='chebwin', at=25.)
    fig4 = plot_2d(snr_array)
    fig4.show()