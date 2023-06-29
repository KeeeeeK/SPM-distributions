import numpy as np
import matplotlib.pyplot as plt

from SPM_distributions.Husimi.husimi import husimi
from SPM_distributions.Wigner.wigner import wigner


def XY_sector(r_mean: float | int, r_range: float | int, r_freq: float | int,
              phi_mean: float | int, phi_range: float | int, phi_freq: float | int):
    r_vals, phi_vals = np.mgrid[r_mean - r_range:r_mean + r_range:complex(0, r_freq),
                                phi_mean - phi_range:phi_mean + phi_range:complex(0, phi_freq)]
    return r_vals * np.cos(phi_vals), r_vals * np.sin(phi_vals)


def XY_rect(x_mean: float | int, x_range: float | int, x_freq: float | int,
            y_mean: float | int, y_range: float | int, y_freq: float | int):
    return np.mgrid[x_mean - x_range:x_mean + x_range:complex(0, x_freq),
                    y_mean - y_range:y_mean + y_range:complex(0, y_freq)]


def plot_3d(X, Y, Z, cmap='inferno'):
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cmap, shade=False)


def plot_from_npz(alpha: complex | float | int, gamma: float | int, file_name: str, cmap='inferno'):
    Z = np.load(file_name)
    freq = Z.shape[0]

    rect_width, rect_height = 5, 120
    X, Y = XY_rect(0, rect_width / 2, freq, 0, rect_height / 2, freq)
    plot_3d(X, Y, Z)



def plot_contourf(X, Y, Z, cmap='inferno'):
    cs = plt.contourf(X, Y, Z, levels=20, cmap=cmap)
    plt.colorbar(cs)
    plt.xlabel(r'Re$\beta-|\alpha|$', loc='right', fontsize='xx-large')
    plt.ylabel(r'Im$\beta$', loc='top', rotation='horizontal', fontsize='xx-large')


def main_part_husimi(alpha: complex | float | int, gamma: float | int, freq: int = 3_00):
    """
    :param alpha: parameter of coherence of initial state
    :param gamma: parameter of non-linearity. In the article it is named \Gamma
    :param freq: number of dots in x and y direction.
    """
    rect_width, rect_height = 2.5, 60

    r_mean = np.abs(alpha)
    phi_mean = - 2 * np.abs(alpha) ** 2 * gamma

    X, Y = XY_rect(0, rect_width / 2, freq, 0, rect_height / 2, freq)
    XY_rotated = (X + r_mean + 1j * Y) * np.exp(1j * phi_mean)
    Z = husimi(np.abs(alpha), np.angle(alpha), np.abs(XY_rotated), np.angle(XY_rotated), gamma, method='2b')
    return X, Y, Z


def main_part_wigner(alpha: complex | float | int, gamma: float | int, freq: int = 150):
    """
    :param alpha: parameter of coherence of initial state
    :param gamma: parameter of non-linearity. In the article it is named \Gamma
    :param freq: number of dots in x and y direction.
    Unfortunately, this method is too slow to show smooth graph in 3d
    """
    rect_width, rect_height = 5, 120

    r_mean = np.abs(alpha)
    phi_mean = - 2 * np.abs(alpha) ** 2 * gamma

    X, Y = XY_rect(0, rect_width / 2, freq, 0, rect_height / 2, freq)
    XY_rotated = (X + r_mean + 1j * Y) * np.exp(1j * phi_mean)
    Z = wigner(np.abs(alpha), np.angle(alpha), np.abs(XY_rotated), np.angle(XY_rotated), gamma, 10**-2)
    print(np.min(Z), np.max(Z))
    # np.save(f'freq{int(freq)}', Z)
    return X, Y, Z



if __name__ == '__main__':
    # if you call main_part_husimi it is rather fast, if you call main_part_wigner it is rather slow
    from time import time

    s = time()

    # plot_3d(*main_part_husimi(2.7 * 1000, 10 ** -6))
    # plot_contourf(*main_part_husimi(2.7 * 1000, 10 ** -6, freq=500))
    # freq=100 ~ 30s, freq=1000 ~ 50min
    # plot_contourf(*main_part_wigner(2.7 * 1000, 10 ** -6, freq=1000))
    plot_from_npz(2.7 * 1000, 10 ** -6, 'freq1000.npy')
    print(time() - s)
    plt.show()
