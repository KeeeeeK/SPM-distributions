import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import scipy as sc

from SPM_distributions.Steepest_descent.constant_phase_curve import constant_phase_curve


def constant_phase_curve_2signs(Z: complex, k_range: np.ndarray, figsize: None | tuple[float, float] = None) -> None:
    """
    Plots constant phase curve for both cases (Gamma>0 and Gamma<0)
    After calling, you should use plt.show or plt.savefig to look at the result
    :param Z: = -2i A * Gamma (designation in the article)
    :param k_range: array of integers.
    Saddle points with numbers from the array k_range will be marked on the graph.
    :param figsize: param for plt.figure. If None, it will be replaced with params used in articule
    """
    steps_params = (0.1, 300, 300)
    x_min, x_max = -20, 20
    y_min, y_max = -10, 15

    plt.figure(figsize=((x_max - x_min) / 2 / 2.54, (y_max - y_min) / 2 / 2.54) if figsize is None else figsize)

    axes = plt.gca()
    axes.set_xlim(x_min, x_max)
    axes.set_ylim(y_min, y_max)

    z = sp.Symbol('z')
    analytic_func = z ** 2 / 2j + 1j * Z * sp.exp(1j * z)
    for k in k_range:
        z_k = 1j * sc.special.lambertw(Z, k=k)
        x_k, y_k = np.real(z_k), np.imag(z_k)
        plt.scatter([np.real(z_k)], [np.imag(z_k)], color='red', marker='o')
        if k != 0:
            axes.annotate(f'$k={k}$', xy=(x_k, y_k), xytext=(x_k + 0.4, y_k + 0.2))
        else:
            axes.annotate(f'$k={k}$', xy=(x_k, y_k), xytext=(x_k + 0.6, y_k - 0.25))
        for gamma_sign in (-1, 1):
            points = constant_phase_curve(z, analytic_func * gamma_sign, (x_k, y_k), steps_params=steps_params)
            plt.plot(*zip(*points), color=f'C{0 if gamma_sign == 1 else 1}')


if __name__ == '__main__':
    constant_phase_curve_2signs(1 + 1j, np.arange(-5, 6, 1))
    plt.show()
    # plt.savefig('constant_phase_curve')
