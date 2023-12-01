import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes

from SPM_distributions.Graphs.plots_3d.XYZ_preparation import XYZ_from_npy
from SPM_distributions.Graphs.plots_3d.main_parts_functions import main_part_husimi, main_part_wigner
from SPM_distributions.Graphs.husimi_connected_2d_plots.F_phi_dependence import set_label

def plot_3d(X, Y, Z, cmap='inferno'):
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cmap, shade=False)


def plot_contourf(X, Y, Z, cmap='inferno'):
    cs = plt.contourf(X, Y, Z, levels=20, cmap=cmap)
    plt.colorbar(cs)
    set_label(plt.gca(), (r'Re$\beta-|\alpha|$', [1.13, -0.03]), (r'Im$\beta$', [0.01, 1.03]))
    # plt.xlabel(, loc='right', fontsize='xx-large')
    # plt.ylabel(r'Im$\beta$', loc='top', rotation='horizontal', fontsize='xx-large')
    return cs.levels


def zoom_plot(X, Y, Z, small_ax_range: tuple[float, float, float, float], zoom: float, cmap: str = 'inferno'):
    x_min, x_max, y_min, y_max = small_ax_range

    def giisa(arr, value):
        # get_index_in_sorted_array
        return np.int_((value - arr[0]) / (arr[-1] - arr[0]) * len(arr))

    Z_s = Z[giisa(X[:, 0], x_min):giisa(X[:, 0], x_max), giisa(Y[0, :], y_min):giisa(Y[0, :], y_max)]
    X_s, Y_s = np.mgrid[x_min:x_max:complex(0, Z_s.shape[0]), y_min:y_max:complex(0, Z_s.shape[1])]

    _small_plot(X_s, Y_s, Z_s, zoom, 20, cmap)


def wigner_small_plot(X_s, Y_s, Z_s, zoom: float, levels, cmap: str = 'inferno'):
    _small_plot(X_s, Y_s, Z_s, zoom, levels, cmap)


def _small_plot(X_s, Y_s, Z_s, zoom, levels, cmap):
    x_min, x_max, y_min, y_max = X_s[0, 0], X_s[-1, 0], Y_s[0, 0], Y_s[0, -1]
    big_ax = plt.gca()

    small_ax = zoomed_inset_axes(big_ax, zoom=zoom, loc='lower right')
    small_ax.set_xlim(x_min, x_max)
    small_ax.set_ylim(y_min, y_max)

    # small_ax.yaxis.get_major_locator().set_params(nbins=1)
    # small_ax.xaxis.get_major_locator().set_params(nbins=1)
    small_ax.tick_params(labelleft=False, labelbottom=False)
    plt.contourf(X_s, Y_s, Z_s, levels=levels, cmap=cmap)
    mark_inset(big_ax, small_ax, loc1=1, loc2=3, fc="none", ec="0.5")


if __name__ == '__main__':
    # if you call main_part_husimi it is rather fast, if you call main_part_wigner it is rather slow.
    # This is because each point in husimi and wigner plots calculates at O(1) and O(alpha) respectively
    from time import time

    target_func = ['husimi', 'wigner'][1]
    s = time()

    if target_func == 'husimi':
        X, Y, Z = main_part_husimi(2.7 * 1000, 10 ** -6, freq=500)
        plot_contourf(X, Y, Z)
        # plot_3d(X, Y, Z)
    elif target_func == 'wigner':
        # freq=100 ~ 30s, freq=1000 ~ 50min
        X, Y, Z = main_part_wigner(2.7 * 1000*1j, 10 ** -6, freq=100, rect_width=1, rect_height=20, save_arr_name=None)
        # X, Y, Z = XYZ_from_npy('freq500_w3_h60.npy')
        levels = plot_contourf(X, Y, Z)
        show_zoomed = False
        # zoom_plot(X, Y, Z, (-0.2, 0.2, -4, 4), 3)
        if show_zoomed is True:
            # X_s, Y_s, Z_s = main_part_wigner(2.7 * 1000, 10 ** -6, freq=300, rect_width=0.4, rect_height=8,
            #                                  save_arr_name='')
            X_s, Y_s, Z_s = XYZ_from_npy('freq500_w3_h60.npy')
            wigner_small_plot(X_s, Y_s, Z_s, 3, levels)
    print(f'finished in {"%.2f" % (time() - s)} seconds')
    plt.show()
