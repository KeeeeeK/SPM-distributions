import numpy as np
from SPM_distributions.Graphs.plots_3d.main_parts_functions import main_part_husimi, main_part_wigner
from SPM_distributions.Graphs.plots_3d.XYZ_preparation import XYZ_from_npy


def integrate_2D(X, Y, Z):
    surface_element = (X[1, 0] - X[0, 0]) * (Y[0, 1] - Y[0, 0])
    return np.sum(Z) * surface_element

def Richardson_integrate_2D(X, Y, Z):
    I1 = integrate_2D(X, Y, Z)
    I2 = integrate_2D(X[::2, ::2], Y[::2, ::2], Z[::2, ::2])
    return I1 + 1/3*(I1-I2)


if __name__ == '__main__':
    target_func = ['wigner', 'husimi', 'wigner_saved'][2]

    if target_func == 'wigner':
        X, Y, Z = main_part_wigner(2.7 * 1000, 10 ** -6, freq=200, rect_width=10, rect_height=120, save_arr_name='')
    elif target_func == 'husimi':
        X, Y, Z = main_part_husimi(2.7 * 1000, 10 ** -6, freq=100, rect_width=10, rect_height=120)
    elif target_func == 'wigner_saved':
        X, Y, Z = XYZ_from_npy('Graphs/plots_3d/freq400_w1_h20.npy')
    else:
        raise KeyError('unknown target func')
    negativity = False
    if negativity is True:
        Z = (np.abs(Z) - Z)/2

    print(np.max(Z[1:,1:]-Z[:-1,:-1]))
    print(np.sum(Z<0)/(Z.shape[0]*Z.shape[1]))
    print(Richardson_integrate_2D(X, Y, Z))
    print(integrate_2D(X, Y, Z))
