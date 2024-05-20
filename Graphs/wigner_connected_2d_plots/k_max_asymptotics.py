import numpy as np
import numba as nb
from SPM_distributions.Husimi.F_normalized import number_type
from SPM_distributions.Graphs.husimi_connected_2d_plots.F_phi_dependence import set_label
import matplotlib.pyplot as plt
from SPM_distributions.Graphs.husimi_connected_2d_plots.plot_steepest_descent \
    import default_figsize, ten_pt_text, fixed_axes

@nb.vectorize('float64(float64, float64, float64, float64)', nopython=True, target_backend='parallel', fastmath=False)
def find_k_max(alpha_abs: number_type, beta_abs: number_type, gamma: number_type, tol=10 ** -2) -> number_type:
    A = 4 * alpha_abs * beta_abs
    # firstly we should know what to sum
    log_tol = np.log(tol)
    left, mid, right = 0, np.pi / 4 / np.abs(gamma), np.pi / 2 / np.abs(gamma)

    def func(k):
        z = A * np.exp(1j * gamma * k)
        return np.real(np.sqrt(z * z + k * k) - k * np.arcsinh(k / z)) - A + \
            alpha_abs * alpha_abs * (1 - np.cos(2 * k * gamma))

    # binary search algorithm to find k_max
    for i in range(np.int_(np.log2(np.pi / 2 / np.abs(gamma)))):
        if func(mid) > log_tol:
            left = mid
        else:
            right = mid
        mid = (left + right) / 2
    k_max = np.int_(mid)
    return k_max


def plot_k_max_alpha_dependence(alpha_arr, beta_arr, gamma_arr, tol, label):
    k_arr = find_k_max(alpha_arr, beta_arr, gamma_arr, tol)
    plt.plot(alpha_arr/10**3, k_arr/10**3, label=label)


if __name__ == '__main__':
    ten_pt_text()
    plt.figure(figsize=(3.15, 3.15))
    alpha_arr = np.arange(3, 3030, 30)
    ax = plt.gca()
    gamma_arr = 2 / alpha_arr ** 2
    for beta_addition, str_addition in ((-2, '-2'), (0, ''), (2, '+2')):
        beta_arr = alpha_arr + beta_addition
        plot_k_max_alpha_dependence(alpha_arr, beta_arr, gamma_arr, 10 ** -2, f'$\\beta = \\alpha {str_addition}$')
    ax.set_xlim(0, 2990/1000)
    ax.set_ylim(0, None)
    set_label(ax, (r'$\alpha, 10^3$', [1, -0.03]), (r'$k_{\max}, 10^3$', [-0.05, 1]))
    plt.legend()
    plt.savefig('1', dpi=500)
