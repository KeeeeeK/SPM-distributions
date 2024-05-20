import warnings
from timeit import timeit
from tqdm import tqdm
import math

import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, wigner

from SPM_distributions.Graphs.husimi_connected_2d_plots.F_phi_dependence import set_label, fixed_axes
from SPM_distributions.Wigner.wigner import wigner as my_wigner
factorial = np.vectorize(math.factorial)


def qutip_psi(N_dim: int, alpha_module: float, gamma: float, alpha_angle: float | None = None) -> Qobj:
    """Returns Qobj from the qutip library with the coordinates of the banana state"""
    alpha_angle = -2 * alpha_module ** 2 * gamma if alpha_angle is None else alpha_angle
    n_small_arr = np.arange(3)
    psi_small_coordinates = np.exp(-alpha_module ** 2 / 2) * alpha_module ** n_small_arr * \
                            np.exp(1j * n_small_arr * gamma + 1j * gamma * n_small_arr * (n_small_arr - 1)) \
                            / np.sqrt(factorial(n_small_arr))
    n_arr = np.arange(3, N_dim)
    psi_big_coordinates = np.exp(
        -alpha_module ** 2 / 2 + 1j * gamma * n_arr * (n_arr - 1) + 1j * alpha_angle * n_arr - np.log(2 * np.pi) / 4
        - 1 / 4 * np.log(n_arr) - n_arr * (np.log(n_arr) / 2 - 1 / 2 - np.log(alpha_module)) - 1 / 24 / n_arr)
    psi_coordinates = np.concatenate((psi_small_coordinates, psi_big_coordinates))
    psi = Qobj(psi_coordinates).conj().unit()
    return psi


def w_cut_by_two_methods(alpha_abs, gamma, alpha_angle, xvec):
    # qutip preparation
    N_dim = int(alpha_abs ** 2 + 5 * alpha_abs)
    psi = qutip_psi(N_dim, alpha_abs, gamma, alpha_angle)
    # Wigner calculation
    my_W = my_wigner(alpha_abs, alpha_angle, xvec, 0, gamma, 10 ** -6)
    qutip_W = 2 * wigner(psi, xvec * np.sqrt(2), 0)[0]  # the formula is different because we count W(alpha),not W(x, p)
    return my_W, qutip_W


def cut_comparison():
    alpha_abs: float = 3.9
    gamma: float = -2.7 / alpha_abs ** 2
    alpha_angle: float = -2 * alpha_abs ** 2 * gamma

    x_mean, x_range, x_freq = alpha_abs, 3., 1000
    xvec = x_mean + np.linspace(-x_range, x_range, x_freq)

    my_W, qutip_W = w_cut_by_two_methods(alpha_abs, gamma, alpha_angle, xvec)
    rescale = 1.3
    ax = fixed_axes(np.min(xvec), np.max(xvec), -0.16, 0.21, figsize=(3.15 * rescale, 3.15 * rescale * 0.7))
    plt.plot(xvec, my_W, label='asymptotic')
    plt.plot(xvec, qutip_W, label='qutip')
    # plt.plot(xvec, my_W - qutip_W, label='diff')
    set_label(ax, (r'$x$', [1.03, -0.0]), (r'$W_{\psi}(x+0i)$', [0.01, 1.03]))
    plt.legend()
    plt.savefig('wigner_cut.png', dpi=500)


def cut_diff_max_comparison():
    alpha_abs_arr = np.arange(6.5, 16, 0.5)
    gamma_arr = 3 / alpha_abs_arr ** 2
    diff_arr = []
    for i in tqdm(range(len(alpha_abs_arr))):
        alpha_abs = alpha_abs_arr[i]
        gamma = gamma_arr[i]
        alpha_angle: float = -2 * alpha_abs ** 2 * gamma
        x_mean, x_range, x_freq = alpha_abs, 3., 1000
        xvec = x_mean + np.linspace(-x_range, x_range, x_freq)
        my_W, qutip_W = w_cut_by_two_methods(alpha_abs, gamma, alpha_angle, xvec)
        diff_arr.append(np.max(np.abs(my_W - qutip_W)))

    y = np.array(diff_arr) * 10 ** 4
    ax = fixed_axes(0, np.max(gamma_arr) * 1.02, 0, np.max(y) * 1.02, figsize=(3.15, 3.15 * 0.7))
    plt.plot(gamma_arr, y, label='diff_max')
    set_label(ax, (r'$\Gamma$', [1.05, -0.03]), (r'$\Delta W\cdot10^4$', [0.01, 1.03]))
    plt.savefig('diff_max', dpi=500)


def timer_qutip(alpha_abs: float, gamma: float, alpha_angle: float, x_vec, y_vec, n_iter: int = 1) -> float:
    N_dim = int(alpha_abs ** 2 + 3.5 * alpha_abs)
    psi = qutip_psi(N_dim, alpha_abs, gamma, alpha_angle).unit()
    return timeit(lambda: 2 * wigner(psi, x_vec * np.sqrt(2), y_vec * np.sqrt(2)).T, number=n_iter) / n_iter


def timer_asymptotic(alpha_abs: float, gamma: float, alpha_angle: float, X, Y, n_iter: int = 1) -> float:
    return timeit(lambda: my_wigner(alpha_abs, alpha_angle, X, Y, gamma, 10 ** -2), number=n_iter) / n_iter


def time_comparison():
    alpha_arr = np.exp(np.arange(np.log(4), np.log(40), 0.07))
    warnings.filterwarnings("error")

    alpha_arr_for_qutip, qutip_stop_calculations = [], False
    asymptotic_time_arr = np.zeros_like(alpha_arr, dtype=float)
    qutip_time_arr = []

    x_freq, y_freq = 100, 100
    n_iter = 1

    for i in tqdm(range(len(alpha_arr))):
        alpha_abs = alpha_arr[i]
        gamma: float = 3 / alpha_abs ** 2
        alpha_angle: float = -2 * alpha_abs ** 2 * gamma
        x_mean, y_mean = alpha_abs, 0
        x_range, y_range = min(4, alpha_abs - 1), min(4, alpha_abs - 1)

        xvec = x_mean + np.linspace(-x_range, x_range, x_freq)
        yvec = y_mean + np.linspace(-y_range, y_range, y_freq)
        X, Y = np.mgrid[x_mean - x_range:x_mean + x_range:complex(0, x_freq),
               y_mean - y_range:y_mean + y_range:complex(0, y_freq)]

        asymptotic_time_arr[i] = timer_asymptotic(alpha_abs, gamma, alpha_angle, X, Y, n_iter=n_iter)
        if qutip_stop_calculations is False:
            try:
                qutip_time_arr.append(timer_qutip(alpha_abs, gamma, alpha_angle, xvec, yvec, n_iter=n_iter))
                alpha_arr_for_qutip.append(alpha_abs)
            except RuntimeWarning:
                qutip_stop_calculations = True
    x_asymp, x_qutip = np.log(alpha_arr), np.log(alpha_arr_for_qutip)
    y_asymp, y_qutip = np.log(asymptotic_time_arr), np.log(qutip_time_arr)

    rescale = 1.1
    ax = fixed_axes(np.min(x_asymp)-0.03, np.max(x_asymp)+0.03, np.min(y_asymp)-0.3, np.max(y_qutip)+0.3,
                    figsize=(3.15 * rescale, 3.15 * rescale * 0.7))
    plt.plot(x_asymp, y_asymp, label='asymptotic')
    plt.scatter([x_qutip[-1]], [y_qutip[-1]], s=60, c='red', marker='X')
    plt.plot(x_qutip, y_qutip, label='qutip')
    set_label(ax, (r'$\ln\alpha$', [1.03, -0.03]), (r'$\ln t/\tau$', [0.01, 1.03]))
    plt.legend()
    plt.savefig('wigner_time', dpi=500)
    plt.show()


if __name__ == '__main__':
    # cut_comparison()
    # cut_diff_max_comparison()
    time_comparison()
