import numpy as np
import numba as nb


@nb.vectorize('float64(float64, float64, float64, float64, float64, float64)',
              nopython=True, target_backend='parallel', fastmath=False)
def wigner(alpha_abs, alpha_arg, beta_abs, beta_arg, gamma, n_sigma):
    """
    alpha is parameter of coherence of initial state
    :param alpha_abs: module of alpha
    :param alpha_arg: angle of alpha
    beta is a parameter of wigner function
    :param beta_abs: module of beta
    :param beta_arg: angle of beta
    :param gamma: parameter of non-linearity. In the article it is named \Gamma
    :param n_sigma: The terms in the limit of small gamma they have a Gaussian distribution. n_sigma is a parameter that
     characterizes the number of standard deviations that will be taken into account when summing.
    :return: the value of wigner function in point beta
    """
    Phi0 = alpha_arg - beta_arg + gamma
    A = 4 * alpha_abs * beta_abs
    norm = 2 * (alpha_abs - beta_abs) ** 2 + A
    # firstly we should know what to sum
    if beta_abs < alpha_abs:
        t_max = np.arccos(beta_abs / alpha_abs)
        k_max = np.int_(t_max / gamma)
        der2f = -1 / (4 * alpha_abs ** 2) - 4 * gamma ** 2 * (alpha_abs ** 2 - beta_abs ** 2) + t_max / 8 * \
                (4 * np.sqrt(alpha_abs ** 2 - beta_abs ** 2) + beta_abs * t_max) / (alpha_abs ** 2 * beta_abs)
    else:
        k_max = 0
        der2f = -1 / (4 * alpha_abs * beta_abs) + 4 * alpha_abs * (alpha_abs - beta_abs) * gamma ** 2
    k_range = np.int_(1 / np.sqrt(-der2f) * n_sigma)
    # now lets sum
    sum_fourier = 0
    for k in nb.prange(-k_max - k_range, k_max + k_range + 1):
        t = np.exp(1j * k * gamma)
        z = A * t
        V_k = np.sqrt(z * z + k * k) - k * np.log(k / z + np.sqrt(1 + (k / z) * (k / z))) - np.log(k * k + z * z) / 4
        sum_fourier += np.exp(1j * k * Phi0 + V_k + alpha_abs * alpha_abs * (1 - t * t) - norm)
    return 2 / np.pi / np.sqrt(2 * np.pi) * np.real(sum_fourier)
