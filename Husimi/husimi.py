import numpy as np
from F_normalized import Fn_sum, Fn_1b, Fn_2b, number_type
from typing import Literal

methods_str = Literal['sum': str, '1b': str, '2b': str]


def husimi(alpha, beta_conj, gamma, method: methods_str = '1b', n_sigma: number_type | None = None):
    match method:
        case 'sum':
            if n_sigma is None:
                n_sigma = 3
                Warning('When using the husimi calculation method by direct summation, it is necessary to specify the '
                        'n_sigma parameter that affects the accuracy of summation.')
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
    from time import time

    alpha = 10 ** 3
    beta_conj = 10 ** 3 * np.exp(1j * np.linspace(0, 2 * np.pi, 10 ** 4))
    gamma = 10 ** -6

    for method in ('sum', '1b', '2b'):
        s = time()
        husimi(alpha, beta_conj, gamma, method=method)
        print(f'calc time of {method}: {time() - s} s')