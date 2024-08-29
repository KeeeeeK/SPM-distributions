import numpy as np
import numba as nb
from SPM_distributions.Husimi.F_normalized import number_type


@nb.vectorize('float64(float64, float64, float64, float64, float64, float64)',
              nopython=True, target='parallel', cache=True)
def fast_wigner(alpha_abs: number_type, alpha_arg: number_type,
           beta_abs: number_type, beta_arg: number_type,
           gamma: number_type, tol: float = 10 ** -2) -> number_type:
    """This is an outdated implementation, but it was the first one that was posted on the arXiv.
    The only fundamental difference is that this implementation gives less accuracy and works a little faster."""
    Phi0 = alpha_arg - beta_arg - gamma
    A = 4 * alpha_abs * beta_abs
    # firstly we should know what to sum
    log_tol = np.log(tol)
    left, mid, right = 1, np.pi / 4 / np.abs(gamma), np.pi / 2 / np.abs(gamma)

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
    # now lets sum
    sum_fourier = 0
    norm = A + 2 * (beta_abs - alpha_abs) ** 2
    for k in nb.prange(1, k_max + 1):
        t = np.exp(1j * k * gamma)
        z = A * t
        V_k = np.sqrt(z * z + k * k) - k * np.arcsinh(k / z) - np.log(k * k + z * z) / 4
        sum_fourier += np.exp(1j * k * Phi0 + V_k + alpha_abs * alpha_abs * (1 - t * t) - norm)
    return 2 / np.pi / np.sqrt(2 * np.pi) * (2 * np.real(sum_fourier) + np.exp(A - norm) / np.sqrt(A))


@nb.vectorize('float64(float64, float64, float64, float64, float64, float64)',
              nopython=True, target_backend='parallel', fastmath=False)
def wigner(alpha_abs: number_type, alpha_arg: number_type,
                    beta_abs: number_type, beta_arg: number_type,
                    gamma: number_type, tol: float = 10 ** -2) -> number_type:
    """
    alpha is parameter of initial coherent state
    :param alpha_abs: module of alpha
    :param alpha_arg: angle of alpha
    beta is a parameter of wigner function
    :param beta_abs: module of beta
    :param beta_arg: angle of beta
    :param gamma: parameter of non-linearity. In the article it is named \Gamma
    :param tol: the accuracy of result values normalized to the first term, MUST BE LESS THAN 1
    The actual accuracy of summation will be tol/(2*alpha_abs*beta_abs)
    :return: the value of wigner function of banana state in point beta
    """
    Phi0 = alpha_arg - beta_arg - gamma
    A = 4 * alpha_abs * beta_abs

    log_tol = np.log(tol)
    R = 2 * alpha_abs ** 2 * np.abs(gamma)
    a = 2 * alpha_abs * (beta_abs - alpha_abs) * np.abs(gamma) + 1 / 4 / R
    if a > 10 * np.sqrt(np.abs(gamma)):
        # most often case
        k_max = alpha_abs * np.sqrt(2 * (-log_tol) / a / R)
    else:  # binary search algorithm to find k_max
        left, mid, right = 1, np.pi / 4 / np.abs(gamma), np.pi / 2 / np.abs(gamma)

        # firstly we should know what to sum
        def func(k):
            z = A * np.exp(1j * gamma * k)
            return np.real(np.sqrt(z * z + k * k) - k * np.arcsinh(k / z)) - A + \
                alpha_abs * alpha_abs * (1 - np.cos(2 * k * gamma))

        for i in range(np.int_(np.log2(np.pi / 2 / np.abs(gamma)))):
            if func(mid) > log_tol:
                left = mid
            else:
                right = mid
            mid = (left + right) / 2
        k_max = np.int_(mid)
    # now lets sum
    sum_fourier = 0
    norm = A + 2 * (beta_abs - alpha_abs) ** 2
    z = A
    zero_term = np.exp(A - norm) / np.sqrt(A) * (1 + 1 / 8 / z + 9 / 128 / z ** 2)
    for k in nb.prange(1, k_max + 1):
        t = np.exp(1j * k * gamma)
        z = A * t
        V_k = np.sqrt(z * z + k * k) - k * np.arcsinh(k / z) - np.log(k * k + z * z) / 4
        a1 = -1 / 8 + 5 / 24 / (1 + (z / k) ** 2)
        a2 = 3 / 128 - 77 / 576 / (1 + (z / k) ** 2) + 385 / 3456 / (1 + (z / k) ** 2) ** 2
        G1 = - a1 / np.sqrt(k ** 2 + z ** 2)
        G2 = a2 * 3 / (k ** 2 + z ** 2)
        sum_fourier += np.exp(1j * k * Phi0 + V_k + alpha_abs * alpha_abs * (1 - t * t) - norm) * (1 + G1 + G2)
    return 2 / np.pi / np.sqrt(2 * np.pi) * (2 * np.real(sum_fourier) + zero_term)


if __name__ == '__main__':
    print(wigner(10 ** 3, 0, 10 ** 3, -2, -10 ** -6, 10 ** -2))
