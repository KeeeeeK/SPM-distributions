import numba as nb
import numpy as np
import scipy as sc

two_pi = 2 * np.pi
half_ln_two_pi = 1 / 2 * np.log(two_pi)

number_type = float | int | np.ndarray


@nb.vectorize('float64(float64, float64, float64, float64)', nopython=True, target_backend='cpu', fastmath=False)
def Fn_sum(r: number_type, phi: number_type, gamma: number_type, n_sigma: number_type) -> nb.float64:
    """
    This is method of calculating F normalized, using direct summation.

    It is expected that r>0, n_sigma>0; phi and gamma are real.
    r, phi, gamma can be numbers or arrays. All arrays used must be of the same length.

    The main idea of this method is the direct summation of the terms in the definition of F. The terms in the limit of
    large r they have a Gaussian distribution. n_sigma is a parameter that characterizes the number of standard
    deviations that will be taken into account when summing.
    """
    # It is bad to use Stirling's formula for the first few terms, so we will sum them up separately
    n_min = max((np.int_(3), np.int_(np.round(r - n_sigma * np.sqrt(r)))))
    n_max = np.int_(np.round(r + n_sigma * np.sqrt(r)))

    sum_ = 0
    for n in nb.prange(n_min, n_max):
        sum_ += np.exp(n - r + n * np.log(r / n) - 0.5 * np.log(n) - 1 / 12 / n - half_ln_two_pi + \
                       1j * np.mod((phi + gamma * n) * n, two_pi))
    first_terms = 1 + r * np.exp(1j * (phi + gamma * 2)) + r ** 2 * np.exp(1j * (2 * phi + gamma * 6)) / 2
    return np.abs(np.exp(-r) * first_terms + sum_)


lambertw = np.vectorize(sc.special.lambertw)


def Fn_1b(r: number_type, phi: number_type, gamma: number_type):
    """
    This is method of calculating F normalized accounting only 1 branch.

    It is expected that r>0, n_sigma>0; phi and gamma are real.
    r, phi, gamma can be numbers or arrays. All arrays used must be of the same length.

    The main idea of this method is to ignore contributions from other branches of the Lambert function except \bar k.
    """
    Z = -2j * r * np.exp(1j * phi) * gamma
    k_bar = -(np.round((np.angle(Z) + (np.abs(Z) + np.pi / 2) * np.sign(gamma)) / (2 * np.pi)))
    z_k = 1j * lambertw(Z, k=np.int_(k_bar))
    f_in_z_k = z_k ** 2 / 2j + z_k
    return np.exp(np.real(f_in_z_k) / (2 * gamma) - r) / np.sqrt(np.abs(1j + z_k))


def Fn_2b(r: number_type, phi: number_type, gamma: number_type):
    """
    This is method of calculating F normalized accounting 2 branches.

    It is expected that r>0, n_sigma>0; phi and gamma are real.
    r, phi, gamma can be numbers or arrays. All arrays used must be of the same length.

    This method is more accurate than Fn_1b, because it takes into account not only the contribution of \bark, but also
    the contribution from the next most important contribution.
    """
    Z = -2j * r * np.exp(1j * phi) * gamma
    k_approx = -(np.angle(Z) + (np.abs(Z) + np.pi / 2) * np.sign(gamma)) / (2 * np.pi)
    k_bar1 = np.floor(k_approx)
    k_bar2 = k_bar1 + 1
    sum_ = 0
    for k_bar in (k_bar1, k_bar2):
        z_k = 1j * sc.special.lambertw(Z, k=np.int_(k_bar))
        f_in_z_k = z_k ** 2 / 2j + z_k
        sum_ += np.exp(f_in_z_k / (2 * gamma) - r) / np.sqrt(1j + z_k)
    return np.abs(sum_)


if __name__ == '__main__':
    # just check that they give the same result
    gamma_ = 0.08
    params = [np.array([1, 2.2])/gamma_, -2 * np.array([1, 2.2]), gamma_]
    print('direct sum:', Fn_sum(*params, 20))
    print('1 branch  :', Fn_1b(*params))
    print('2 branches:', Fn_2b(*params))
