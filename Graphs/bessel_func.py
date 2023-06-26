import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from SPM_distributions.Husimi.F_normalized import number_type
from SPM_distributions.Graphs.steepest_descent_graphs import fixed_axes
from SPM_distributions.Graphs.F_phi_dependence import set_label


def V_k(z: complex | number_type, k: int | np.ndarray):
    return np.sqrt(z * z + k * k) - k * np.arcsinh(k / z * np.sign(np.real(z))) - np.log(k * k + z * z) / 4


def plot_ln_I_k(A_abs: float | int, gamma: float | int, K_max: int = 3):
    k = np.arange(-np.int_(K_max * np.pi / gamma), np.int_(K_max * np.pi / gamma) + 1)
    ln_I_k = np.real(V_k(2 * A_abs * np.exp(-1j * gamma * k), k)) - 2 * A_abs

    axes = fixed_axes(np.min(k), np.max(k), np.min(ln_I_k), 0, figsize=None)

    def formatter(val):
        t = val * gamma / np.pi
        str_add = r'$\dfrac{\pi}{\Gamma}$'
        if np.isclose(t, 0):
            return '0'
        if np.isclose(t, 1):
            return str_add
        if np.isclose(t, -1):
            return '-' + str_add
        return r'{:.0g}'.format(t) + str_add

    axes.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda val, pos: formatter(val)))
    axes.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=np.pi / gamma))
    set_label(axes, r'$k$', r'Re$V_k(2|A|e^{-i\Gamma k})-2|A|$')
    for i in range(K_max):
        axes.vlines((i + 1 / 2) * np.pi / gamma, np.min(ln_I_k), 100, linewidth=1, colors='black')
        axes.vlines(-(i + 1 / 2) * np.pi / gamma, np.min(ln_I_k), 100, linewidth=1, colors='black')

        if i == 0:
            axes.text(0, np.min(ln_I_k) * 0.97, '$K=0$', ha='center')
        else:
            axes.text(i * np.pi / gamma, np.min(ln_I_k) * 0.97, f'$K={i}$', ha='center')
            axes.text(-i * np.pi / gamma, np.min(ln_I_k) * 0.97, f'$K={-i}$', ha='center')
    plt.scatter(k, ln_I_k, s=1)


def plot_ln_summand_K0(alpha_abs: float | int, beta_abs: float | int, gamma: float | int):
    k_range = 3200
    k = np.arange(-k_range, k_range + 1)
    ln_I_k = np.real(V_k(4 * alpha_abs * beta_abs * np.exp(-1j * gamma * k), k)) - 4 * alpha_abs * beta_abs
    ln_summand = ln_I_k + alpha_abs ** 2 * (1 - np.cos(2 * k * gamma))

    axes = fixed_axes(np.min(k), np.max(k), np.min(ln_summand),
                      max(np.max(ln_summand) + (np.max(ln_summand) - np.min(ln_summand)) / 100, 0), figsize=None)
    # axes = plt.gca()
    # axes.vlines(np.arccos(beta_abs / alpha_abs) / gamma / 4, np.min(ln_summand), np.max(ln_summand), linewidth=1,
    #             colors='black')
    set_label(axes, r'$k$', r'')
    plt.scatter(k, ln_summand, s=0.1)
    plt.scatter(k, ln_summand+10**6, s=10, c='C0' ,
                label=r'$\ln\left(I_k(4|\alpha\beta^*|e^{-i\Gamma k})\right)+|\alpha|^2\left(1-\cos(2k\Gamma)\right)$')
    plt.scatter(k, ln_I_k, s=0.1)
    plt.scatter(k, ln_I_k+10**6, s=10, c='C1', label=r'$\ln\left(I_k(4|\alpha\beta^*|e^{-i\Gamma k})\right)$')
    plt.legend()


if __name__ == '__main__':
    # plot_ln_I_k(300, 0.01)
    plot_ln_summand_K0(10 ** 2, 10 ** 2 - 8, 10 ** -4)
    # plot_ln_summand_K0(10 ** 3, 10 ** 3, 10 ** -6)
    plt.show()
