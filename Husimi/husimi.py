from typing import Literal
import numpy as np
from SPM_distributions.Husimi.F_normalized import Fn_sum, Fn_1b, Fn_2b, number_type

methods_str = Literal['sum': str, '1b': str, '2b': str]


def husimi(alpha_abs: number_type, alpha_arg: number_type,
           beta_abs: number_type, beta_arg: number_type,
           gamma: number_type, method: methods_str = '1b', n_sigma: number_type = 3) -> number_type:
    """
    alpha is parameter of coherence of initial state
    :param alpha_abs: module of alpha
    :param alpha_arg: angle of alpha
    beta is a parameter of wigner function
    :param beta_abs: module of beta
    :param beta_arg: angle of beta
    :param gamma: parameter of non-linearity. In the article it is named \\Gamma
    :param method: the method that will be used to find the values of the F normalized.
     It should be on of 'sum' (direct summation), '1b' (1 branch), '2b' (2 branches).
     For further details see the documentation in the F_normalized.py file
    :param n_sigma: parameter that will be used only if method == 'sum'. More details in Fn_sum documentation
    :return: the value of husimi function in point beta
    """
    match method:
        case 'sum':
            return np.exp(-(alpha_abs - beta_abs) ** 2) / np.pi * \
                np.abs(Fn_sum(alpha_abs * beta_abs, alpha_arg - beta_arg - gamma, gamma, n_sigma)) ** 2
        case '1b':
            return np.exp(-(alpha_abs - beta_abs) ** 2) / np.pi * \
                np.abs(Fn_1b(alpha_abs * beta_abs, alpha_arg - beta_arg - gamma, gamma)) ** 2
        case '2b':
            return np.exp(-(alpha_abs - beta_abs) ** 2) / np.pi * \
                np.abs(Fn_2b(alpha_abs * beta_abs, alpha_arg - beta_arg - gamma, gamma)) ** 2
        case _:
            raise ValueError('Unknown method')


if __name__ == '__main__':
    # This is a simple speed test of various methods.
    # Using methods using branches of the Lambert function increases the speed by more than 100 times.
    from time import time

    alpha = 10 ** 3
    beta_abs = 10 ** 3
    beta_arg = np.linspace(0, 2 * np.pi, 10 ** 4)
    gamma = 10 ** -6

    for method in ('2b', '1b', 'sum'):
        s = time()
        husimi(alpha, 0, beta_abs, beta_arg, gamma, method=method)
        print(f'calc time of {method}: {time() - s} s')
