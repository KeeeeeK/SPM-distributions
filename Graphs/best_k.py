import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Sequence


def _best_k_slow(Z: int | float | complex, k_sign: -1 | 1) -> int:
    """This is a rather slow algorithm for finding \bar k at some Z, which makes the main contribution to the sum over
    the branches of the Lambert function"""
    k_mean_abs, k_range = np.int_(np.abs(Z) / (2 * np.pi)), 2
    k_arr = np.arange(np.max((0, k_mean_abs - k_range)), k_mean_abs + k_range + 1) * k_sign
    z_k = np.array([1j * sc.special.lambertw(Z, k=k) for k in k_arr])
    f_in_z_k = z_k ** 2 / 2j + z_k
    k_max = k_arr[np.argmax(np.real(f_in_z_k * (-k_sign)))]
    return k_max


def plot_best_k(x_step_params: tuple[float, float, int],
                y_step_params: tuple[float, float, int],
                k_sign: -1 | 1, alpha: int | float = 1) -> None:
    """
    The method paints the complex plane in different colors depending on how the optimal \bar k is at the corresponding
     point
    :param x_step_params: (x_min, x_max, n_dots). similarly for y_step_params
    :param alpha: opacity of the layer. Set a value other than 1 if you need to display multiple intersecting areas.
    """
    # print(tuple(it.product(np.arange(x_step_params[2]), np.arange(y_step_params[2]))))
    X, Y = np.mgrid[x_step_params[0]:x_step_params[1]:complex(0, x_step_params[2]),
           y_step_params[0]:y_step_params[1]:complex(0, y_step_params[2])]
    K = np.zeros((x_step_params[2], y_step_params[2]))
    for i in tqdm(range(x_step_params[2])):
        for j in range(y_step_params[2]):
            K[i, j] = _best_k_slow(X[i, j] + 1j * Y[i, j], k_sign)

    ax = plt.gca()
    ax.pcolor(X, Y, K, cmap='cool', shading='nearest', alpha=alpha)


def plot_difference_less_eps(eps: float,
                             x_step_params: tuple[float, float, int],
                             y_step_params: tuple[float, float, int],
                             k_sign: -1 | 1, alpha: int | float = 1) -> None:
    """
    The method displays that small area of Z values in which the relative difference of contribution from $\bar k$ and
    $\bar k + 1$ or $\bar k -1$ is less than eps.
    :param x_step_params: (x_min, x_max, n_dots). similarly for y_step_params
    :param alpha: opacity of the layer. Set a value other than 1 if you need to display multiple intersecting areas.
    """
    X, Y = np.mgrid[x_step_params[0]:x_step_params[1]:complex(0, x_step_params[2]),
           y_step_params[0]:y_step_params[1]:complex(0, y_step_params[2])]
    Z = X + 1j * Y
    diff = np.zeros((x_step_params[2], y_step_params[2]))
    for i in tqdm(range(x_step_params[2])):
        for j in range(y_step_params[2]):
            k_mean_abs, k_range = np.int_(
                (np.abs(Z[i, j]) + np.abs(np.angle(Z[i, j]) - np.pi / 2 * k_sign)) / (2 * np.pi)), 3
            k_arr = np.arange(np.max((0, k_mean_abs - k_range)), k_mean_abs + k_range + 1) * k_sign
            z_k = np.array([1j * sc.special.lambertw(Z[i, j], k=k) for k in k_arr])
            f_in_z_k = z_k ** 2 / 2j + z_k
            vals = np.real(f_in_z_k * (-k_sign))

            if np.min(np.abs(vals[1:] - vals[:-1])) < eps:
                diff[i, j] = -1
            else:
                diff[i, j] = _best_k_slow(X[i, j] + 1j * Y[i, j], k_sign)

    ax = plt.gca()
    ax.pcolor(X, Y, diff,
              cmap='cool', shading='nearest', alpha=alpha)


def plot_annotations_for_k_bar(xy_tuples: Sequence[tuple[float | int, float | int]], k_sign: -1 | 1) -> None:
    """
    :param xy_tuples: array of positions of points where annotations will be written.
    If a point (x, y) is specified, it will be displayed at the position (x, y*k_sign)
    """
    axes = plt.gca()
    for k, (x, y) in enumerate(xy_tuples):
        axes.annotate(r'$\bar{k}=' + str(k * k_sign) + r'$', xy=(x, y * k_sign), xytext=(x + 0.4, y * k_sign + 0.2))


if __name__ == '__main__':
    Z_range, freq, k_sign = 11, 100, 1

    plot_best_k((-Z_range, Z_range, freq), (-Z_range, Z_range, freq), k_sign, alpha=1)
    plot_difference_less_eps(0.02, (-Z_range, Z_range, freq), (-Z_range, Z_range, freq), k_sign, alpha=1)
    plot_annotations_for_k_bar(((0, 0), (-1.8, -2.54), (-6, -6.5), (-10, -10)), k_sign)

    plt.show()
