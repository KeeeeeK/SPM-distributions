from typing import Literal
import numpy as np
from F_normalized import Fn_sum, Fn_1b, Fn_2b, number_type

methods_str = Literal['sum': str, '1b': str, '2b': str]


def husimi(alpha: complex | number_type,
           beta_conj: complex | number_type,
           gamma: number_type,
           method: methods_str = '1b', n_sigma: number_type = 3) -> number_type:
    """
    :param alpha: parameter of coherence of initial state
    :param beta_conj: conjugated beta, where beta is a parameter of husimi function
    :param gamma: parameter of non-linearity. In the article it is named \Gamma
    :param method: the method that will be used to find the values of the F normalized.
     It should be on of 'sum' (direct summation), '1b' (1 branch), '2b' (2 branches).
     For further details see the documentation in the F_normalized.py file
    :param n_sigma: parameter that will be used only if method == 'sum'.
    :return: the value of husimi function in point beta
    """
    match method:
        case 'sum':
            return np.exp(-(np.abs(alpha) - np.abs(beta_conj)) ** 2) / np.pi * \
                np.abs(Fn_sum(np.abs(alpha * beta_conj), 2 * gamma + np.angle(alpha * beta_conj), -gamma, n_sigma)) ** 2
        case '1b':
            return np.exp(-(np.abs(alpha) - np.abs(beta_conj)) ** 2) / np.pi * \
                np.abs(Fn_1b(np.abs(alpha * beta_conj), 2 * gamma + np.angle(alpha * beta_conj), -gamma)) ** 2
        case '2b':
            return np.exp(-(np.abs(alpha) - np.abs(beta_conj)) ** 2) / np.pi * \
                np.abs(Fn_2b(np.abs(alpha * beta_conj), 2 * gamma + np.angle(alpha * beta_conj), -gamma)) ** 2
        case _:
            raise ValueError('Unknown method')


if __name__ == '__main__':
    # This is a simple speed test of various methods.
    # Using methods using branches of the Lambert function increases the speed by more than 100 times.
    from time import time

    alpha = 10 ** 3
    beta_conj = 10 ** 3 * np.exp(1j * np.linspace(0, 2 * np.pi, 10 ** 4))
    gamma = 10 ** -6

    for method in ('1b', '2b', 'sum'):
        s = time()
        husimi(alpha, beta_conj, gamma, method=method)
        print(f'calc time of {method}: {time() - s} s')
