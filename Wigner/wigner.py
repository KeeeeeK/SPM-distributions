import numpy as np
import numba as nb
from SPM_distributions.Husimi.F_normalized import number_type


@nb.vectorize('float64(float64, float64, float64, float64, float64, float64)',
              nopython=True, target_backend='parallel', fastmath=False)
def wigner(alpha_abs: number_type, alpha_arg: number_type,
           beta_abs: number_type, beta_arg: number_type,
           gamma: number_type, tol=10 ** -3) -> number_type:
    """
    alpha is parameter of coherence of initial state
    :param alpha_abs: module of alpha
    :param alpha_arg: angle of alpha
    beta is a parameter of wigner function
    :param beta_abs: module of beta
    :param beta_arg: angle of beta
    :param gamma: parameter of non-linearity. In the article it is named \Gamma
    :param tol: the accuracy, less than 1
    :return: the value of wigner function in point beta
    """
    Phi0 = alpha_arg - beta_arg + gamma
    A = 4 * alpha_abs * beta_abs
    # firstly we should know what to sum
    log_tol = np.log(tol)
    left, mid, right = 0, np.pi / 4 / gamma, np.pi / 2 / gamma

    def func(k):
        z = A * np.exp(-1j * gamma * k)
        return np.real(np.sqrt(z * z + k * k) - k * np.arcsinh(k / z)) - A + alpha_abs * alpha_abs * (
                    1 - np.cos(2 * k * gamma))

    # binary search algorithm
    for i in range(np.int_(np.log2(np.pi / 2 / gamma))):
        if func(mid) > log_tol:
            left = mid
        else:
            right = mid
        mid = (left + right) / 2
    k_max = np.int_(mid)
    # now lets sum
    sum_fourier = 0
    norm = A + (beta_abs - alpha_abs) ** 2
    for k in nb.prange(1, k_max + 1):
        t = np.exp(-1j * k * gamma)
        z = A * t
        V_k = np.sqrt(z * z + k * k) - k * np.arcsinh(k / z) - np.log(k * k + z * z) / 4
        sum_fourier += np.exp(1j * k * Phi0 + V_k + alpha_abs * alpha_abs * (1 - t * t) - norm)
    return 2 / np.pi / np.sqrt(2 * np.pi) * (2 * np.real(sum_fourier) - np.exp(A - norm) / np.sqrt(A))


if __name__ == '__main__':
    print(wigner(10 ** 3, 0, 10 ** 3, -2, 10 ** -6, 10 ** -2))
