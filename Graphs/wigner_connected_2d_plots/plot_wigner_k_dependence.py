import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from SPM_distributions.Husimi.F_normalized import number_type
from SPM_distributions.Graphs.husimi_connected_2d_plots.plot_steepest_descent import fixed_axes
from SPM_distributions.Graphs.husimi_connected_2d_plots.F_phi_dependence import set_label


def ten_pt_text():
    mpl.rcParams.update({'font.size': 110/15})

def V_k(z: complex | number_type, k: int | np.ndarray):
    return np.sqrt(z * z + k * k) - k * np.arcsinh(k / z * np.sign(np.real(z))) - np.log(k * k + z * z) / 4


def plot_ln_I_k(A_abs: float | int, gamma: float | int, K_max: int = 3):
    k = np.arange(-np.int_(K_max * np.pi / gamma), np.int_(K_max * np.pi / gamma) + 1)
    ln_I_k = np.real(V_k(2 * A_abs * np.exp(-1j * gamma * k), k)) - 2 * A_abs

    axes = fixed_axes(np.min(k), np.max(k), np.min(ln_I_k), 0, figsize=(3.15,3.15))

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
    set_label(axes, (r'$k$', [1.04, 0]), (r'Re$V_k(2|A|e^{-i\Gamma k})-2|A|$', [0.07, 1.03]))
    for i in range(K_max):
        axes.vlines((i + 1 / 2) * np.pi / gamma, np.min(ln_I_k), 100, linewidth=1, colors='black')
        axes.vlines(-(i + 1 / 2) * np.pi / gamma, np.min(ln_I_k), 100, linewidth=1, colors='black')

        if i == 0:
            axes.text(0, np.min(ln_I_k) * 0.97, '$K=0$', ha='center')
        else:
            axes.text(i * np.pi / gamma, np.min(ln_I_k) * 0.97, f'$K={i}$', ha='center')
            axes.text(-i * np.pi / gamma, np.min(ln_I_k) * 0.97, f'$K={-i}$', ha='center')
    plt.scatter(k, ln_I_k, s=1)


def plot_ln_summand_K0(alpha_abs: float | int, delta_beta_abs: list[float | int], gamma: float | int, k_range: int):
    beta_abs_arr = alpha_abs - np.array(delta_beta_abs)
    k = np.arange(-k_range, k_range + 1)
    axes = fixed_axes(np.min(k), np.max(k), -30, 0, figsize=(3.15,3.15))
    set_label(axes, (r'$k$', [1.04, 0]),
      (r'$\ln\left(I_k(4|\alpha\beta^*|e^{i\Gamma k})\right)+|\alpha|^2\left(1-\cos(2k\Gamma)\right)$', [0.27, 1.03]))
    for n, beta_abs in enumerate(beta_abs_arr):
        ln_I_k = np.real(V_k(4 * alpha_abs * beta_abs * np.exp(1j * gamma * k), k)) - 1 / 2 * np.log(2 * np.pi)
        ln_summand = ln_I_k + alpha_abs ** 2 * (1 - np.cos(2 * k * gamma)) - 4 * alpha_abs * beta_abs
        plt.scatter(k, ln_summand, c=f'C{n}', s=0.1)
        plt.scatter([], [], s=10, c=f'C{n}', label=f'$\\beta=\\alpha-{delta_beta_abs[n]}$')
    # axes = plt.gca()

    # axes.hlines(np.log(10 ** -3), np.min(k), np.max(k), linewidth=1, colors='black')

    # plt.scatter(k, ln_I_k, s=0.1)
    # plt.scatter(k, ln_I_k + 10 ** 6, s=10, c='C1', label=r'$\ln\left(I_k(4|\alpha\beta^*|e^{i\Gamma k})\right)$')
    plt.legend()


def plot_zoomed_ln_summand_K0(alpha_abs: float | int, beta_abs: float | int, gamma: float | int, k_range: int):
    k = np.arange(-k_range, k_range + 1)
    ln_I_k = np.real(V_k(4 * alpha_abs * beta_abs * np.exp(1j * gamma * k), k)) - 1 / 2 * np.log(2 * np.pi)
    ln_summand = ln_I_k + alpha_abs ** 2 * (1 - np.cos(2 * k * gamma)) - 4 * alpha_abs * beta_abs

    axes = fixed_axes(np.min(k), np.max(k), np.min(ln_summand),
                      np.max(ln_summand) + (np.max(ln_summand) - np.min(ln_summand)) / 100, figsize=(3.15,3.15))
    axes.hlines(np.log(10 ** -7), np.min(k), np.max(k), linewidth=1, colors='black')
    set_label(axes, (r'$k$', [1.04, 0]), (r'', [-0.01, 1.03]))
    plt.scatter(k, ln_summand, s=0.1)
    plt.scatter(k, ln_summand + 10 ** 6, s=10, c='C0',
                label=r'$\ln\left(I_k(4|\alpha\beta^*|e^{i\Gamma k})\right)+|\alpha|^2\left(1-\cos(2k\Gamma)\right)$')
    # plt.scatter(k, ln_I_k, s=0.1)
    # plt.scatter(k, ln_I_k + 10 ** 6, s=10, c='C1', label=r'$\ln\left(I_k(4|\alpha\beta^*|e^{i\Gamma k})\right)$')
    plt.legend()


if __name__ == '__main__':
    ten_pt_text()
    target_func = ['ln_I_k', 'unusual_ln_summand_K0', 'zoomed_ln_summand_K0'][0]
    if target_func == 'ln_I_k':
        plot_ln_I_k(300, 0.01)
    elif target_func == 'unusual_ln_summand_K0':
        plot_ln_summand_K0(10 ** 2, [8, 2], -10 ** -4, 3200)
    elif target_func == 'zoomed_ln_summand_K0':
        plot_zoomed_ln_summand_K0(10 ** 3, 10 ** 3 - 66.9873, 10 ** -6, 1000)
    plt.savefig('1', dpi=500)
