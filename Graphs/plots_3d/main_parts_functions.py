import numpy as np

from SPM_distributions.Husimi.husimi import husimi
from SPM_distributions.Wigner.wigner import wigner
from SPM_distributions.Graphs.plots_3d.XYZ_preparation import XY_rect

def main_part_husimi(alpha: complex | float | int, gamma: float | int, freq: int = 3_00,
                     rect_width: float | int = 2.5, rect_height: float | int = 60):
    """
    :param alpha: parameter of coherence of initial state
    :param gamma: parameter of non-linearity. In the article it is named \Gamma
    :param freq: number of dots in x and y direction.
    """

    r_mean = np.abs(alpha)
    phi_mean = 2 * np.abs(alpha) ** 2 * gamma

    X, Y = XY_rect(0, rect_width / 2, freq, 0, rect_height / 2, freq)
    XY_rotated = (X + r_mean + 1j * Y) * np.exp(1j * phi_mean)
    Z = husimi(np.abs(alpha), np.angle(alpha), np.abs(XY_rotated), np.angle(XY_rotated), gamma, method='2b')
    return X, Y, Z


def main_part_wigner(alpha: complex | float | int, gamma: float | int, freq: int = 150, save_arr_name: str | None = None,
                     rect_width: float | int = 5, rect_height: float | int = 120):
    """
    :param alpha: parameter of coherence of initial state
    :param gamma: parameter of non-linearity. In the article it is named \Gamma
    :param freq: number of dots in x and y direction.
    Unfortunately, this method is too slow to show smooth graph in 3d
    """

    r_mean = np.abs(alpha)
    phi_mean = 2 * np.abs(alpha) ** 2 * gamma + np.angle(alpha)

    X, Y = XY_rect(0, rect_width / 2, freq, 0, rect_height / 2, freq)
    XY_rotated = (X + r_mean + 1j * Y) * np.exp(1j * phi_mean)
    Z = wigner(np.abs(alpha), np.angle(alpha), np.abs(XY_rotated), np.angle(XY_rotated), gamma, 10 ** -2)
    print(f'Wigner results: min = {np.min(Z)}, max = {np.max(Z)}')
    if save_arr_name is not None:
        import sys
        # rect_width and rect_height recommended to be integer to make file more readable
        with open(f'{sys.path[0]}/freq{int(freq)}_w{int(rect_width)}_h{int(rect_height)}{save_arr_name}.npy', 'wb') as f:
            np.save(f, Z)
            np.save(f, np.array([rect_width, rect_height]))
    return X, Y, Z

