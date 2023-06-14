import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import sympy as sp
import scipy as sc

from SPM_distributions.Steepest_descent.constant_phase_curve import constant_phase_curve


def z_k_position(Z: complex = 1 + 1j,
                 k_range: np.ndarray = np.arange(-6, 6 + 1, 1),
                 figsize: None | tuple[float, float] = None):
    x_min, x_max = -22, 22
    y_min, y_max = -10, 10
    axes = _fixed_axes(x_min, x_max, y_min, y_max, figsize)
    _arrows(axes, x_min, x_max, y_min, y_max, 0.0005)  # bold axes with arrows

    # axis tuning
    axes.spines['bottom'].set_position('center')
    axes.spines[['left', 'top', 'right']].set_visible(False)
    plt.yticks([])
    axes.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda val, pos: '{:.0g}$\pi$'.format(val / np.pi) if val != 0 else ''))
    axes.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=np.pi))

    for k in k_range:
        # plotting points z_k
        z_k = 1j * sc.special.lambertw(Z, k=k)
        x_k, y_k = np.real(z_k), np.imag(z_k)
        plt.scatter([x_k], [y_k], color='red', marker='o', s=14)
        # plotting borders of area where z_k may appear
        eps = 10 ** -3
        if k > 0:
            axes.annotate(f'$k={k}$' if k < 4 else r'', xy=(0, y_max - 2), xytext=(2 * np.pi * k - 2, y_max - 2))
            _draw_border_of_z_k_area(np.linspace(2 * k * np.pi + eps, 2 * (k + 1 / 2) * np.pi - eps, 50))
        elif k < 0:
            axes.annotate(f'$k={k}$' if k > -4 else r'', xy=(0, y_max - 2), xytext=(2 * np.pi * k - 1, y_max - 2))
            _draw_border_of_z_k_area(np.linspace(2 * (k + 1 / 2) * np.pi + eps, 2 * (k + 1) * np.pi - eps, 50))
        else:  # k==0
            axes.annotate(f'$k=0$', xy=(0, y_max - 2), xytext=(-1, y_max - 2))
            _draw_border_of_z_k_area(np.linspace(2 * k * np.pi + eps, 2 * (k + 1 / 2) * np.pi - eps, 50))
            plt.plot([0, 0], [y_min, -1], color='black', linestyle='--', linewidth=2)


def _draw_border_of_z_k_area(eta: np.ndarray | float):
    z_border = 1j * (-eta / np.tan(eta) - eta * 1j)
    x_border, y_border = np.real(z_border), np.imag(z_border)
    plt.plot(x_border, y_border, color='black', linestyle='--', linewidth=2)


def integration_contour(Z: complex = 1 + 1j,
                        k_range: np.ndarray = np.arange(-5, 1, 1),
                        figsize: None | tuple[float, float] = None):
    steps_params = (0.1, 500, 300)
    big_step = 5  # the frequency of the arrows displaying the direction of integration
    x_min, x_max = -40, 40
    y_min, y_max = -10, 35
    axes = _fixed_axes(x_min, x_max, y_min, y_max, figsize)
    path_marker, markersize = _arrow_path(), 10
    plt.axis('off')
    _arrows(axes, x_min, x_max, y_min, y_max, 0.0002)  # bold axes with arrows

    # plotting main part of new contour
    z = sp.Symbol('z')
    analytic_func = z ** 2 / 2j + 1j * Z * sp.exp(1j * z)
    if 0 not in k_range:
        raise ValueError("0 should be in k_range anyway")
    for k in k_range:
        # plotting points z_k
        z_k = 1j * sc.special.lambertw(Z, k=k)
        x_k, y_k = np.real(z_k), np.imag(z_k)
        plt.scatter([x_k], [y_k], color='red', marker='o', s=14)
        annotation = f'$k={k}$' if abs(k) <= 3 else r'$\dots$'
        axes.annotate(annotation, xy=(x_k, y_k), xytext=(x_k + 0.5, y_k + 0.08) if k != 0 else (x_k + 0.6, y_k - 0.25))
        # finding curve
        points = constant_phase_curve(z, analytic_func, (x_k, y_k), steps_params=steps_params)
        # saving some params for I_+ and I_-. For k==0 we should also limit range of variation of y
        if k == np.min(k_range):
            R = points[-1][0]
        elif k == 0:
            y_max_I_minus = _intersection_with_vertical_line(points[:steps_params[1]], -R)
            points = points[points[:, 1] < y_max_I_minus]
        # plotting curve
        plt.plot(*zip(*points), color='C2')
        # showing direction of integration
        num_step = int(big_step / (2 * steps_params[0]))  # the amount of halves of small steps in big step
        for point, next_point in zip(points[num_step::num_step * 2], points[num_step + 1::num_step * 2]):
            theta = np.angle((next_point[0] - point[0] + 1j * (next_point[1] - point[1])))
            marker = mpl.markers.MarkerStyle(path_marker, 'full', mpl.transforms.Affine2D().rotate(theta))
            plt.plot([point[0]], [point[1]], color='C2', marker=marker, markersize=markersize)

    # showing I_+
    y_I_plus = np.arange(y_min - big_step // 2, 0, big_step)
    marker_up = mpl.markers.MarkerStyle(path_marker, 'full', mpl.transforms.Affine2D().rotate_deg(90))
    plt.plot(np.tile(R, len(y_I_plus)), y_I_plus, color='C2', marker=marker_up, markersize=markersize)
    plt.plot([R, R], [y_I_plus[-1], 0], color='C2')
    # showing I_-
    y_I_minus = np.arange(big_step / 2, y_max_I_minus, big_step)
    plt.plot(np.tile(-R, len(y_I_minus)), y_I_minus, color='C2', marker=marker_up, markersize=markersize)
    plt.plot([-R, -R], [y_max_I_minus, 0], color='C2')

    # plotting old contour
    x_horizontal = np.arange(-R + big_step / 6, R, big_step)
    marker_right = mpl.markers.MarkerStyle(path_marker, 'full')
    plt.plot(x_horizontal, np.zeros(len(x_horizontal)), color='C0', marker=marker_right, markersize=markersize)
    plt.plot((-R, R), (0, 0), color='C0')

    # annotate the points (-R, 0) and (R, 0)
    plt.scatter((-R, R), (0, 0), color='black', s=14)
    axes.annotate('(-R, 0)', xy=(-R, 0), xytext=(-R + 0.5, 0.23))
    axes.annotate('(R, 0)', xy=(R, 0), xytext=(R + 0.5, 0.23))
    # the legend is written with fake dots, which will not be visible on the graph
    plt.plot((x_max * 2,), (y_max * 2,), color='C0', marker=marker_right, markersize=markersize,
             label='initial integration contour')
    plt.plot((x_max * 2,), (y_max * 2,), color='C2', marker=marker_right, markersize=markersize,
             label='new integration contour')
    plt.scatter((x_max * 2,), (y_max * 2,), color='red', marker='o', s=14, label='saddle points')
    plt.legend()


def constant_phase_curve_2signs(Z: complex = 1 + 1j,
                                k_range: np.ndarray = np.arange(-5, 6, 1),
                                figsize: None | tuple[float, float] = None) -> None:
    """
    Plots constant phase curve for both cases (Gamma>0 and Gamma<0)
    After calling, you should use plt.show or plt.savefig to look at the result
    :param Z: = -2i A * Gamma (designation in the article)
    :param k_range: array of integers.
    Saddle points with numbers from the array k_range will be marked on the graph.
    :param figsize: param for plt.figure. If None, it will be replaced with params used in article
    """
    steps_params = (0.1, 300, 300)
    x_min, x_max = -20, 20
    y_min, y_max = -10, 15
    axes = _fixed_axes(x_min, x_max, y_min, y_max, figsize)
    _arrows(axes, x_min, x_max, y_min, y_max, 0.00001)  # bold axes with arrows

    z = sp.Symbol('z')
    analytic_func = z ** 2 / 2j + 1j * Z * sp.exp(1j * z)
    for k in k_range:
        z_k = 1j * sc.special.lambertw(Z, k=k)
        x_k, y_k = np.real(z_k), np.imag(z_k)
        plt.scatter([x_k], [y_k], color='red', marker='o')
        axes.annotate(f'$k={k}$', xy=(x_k, y_k), xytext=(x_k + 0.4, y_k + 0.2) if k != 0 else (x_k + 0.6, y_k - 0.25))
        for gamma_sign in (-1, 1):
            points = constant_phase_curve(z, analytic_func * gamma_sign, (x_k, y_k), steps_params=steps_params)
            plt.plot(*zip(*points), color=f'C{0 if gamma_sign == 1 else 1}')


def _arrows(axes, x_min, x_max, y_min, y_max, dy: float) -> None:
    # dy is crutch... to make right arrow slightly lower
    arrowprops = dict(arrowstyle=mpl.patches.ArrowStyle.CurveB(head_length=1), color='black')
    p = - y_min / (y_max - y_min)
    axes.annotate('', xy=(1.05, p - dy), xycoords='axes fraction', xytext=(0.99, p - dy), arrowprops=arrowprops)
    axes.annotate('', xy=(0.5, 1.05), xycoords='axes fraction', xytext=(0.5, 0.99), arrowprops=arrowprops)
    axes.annotate(r'Re$(z)$', xy=(x_max, 0), xytext=(x_max + 0.03 * (x_max - x_min), 0.02 * (y_max - y_min)))
    axes.annotate(r'Im$(z)$', xy=(0, y_max), xytext=(0.01 * (x_max - x_min), y_max + 0.02 * (y_max - y_min)))
    axes.hlines(0, x_min, x_max, linewidth=1, colors='black')
    axes.vlines(0, y_min, y_max, linewidth=1, colors='black')
    axes.spines[['right', 'top']].set_visible(False)


def _arrow_path():
    """Makes mpl.path.Path object which looks like an arrow. Use to show the direction of integration contour"""
    vertexes = [(2., 0.), (-1., 1.), (0., 0.), (-1., -1.), (2., 0.)]
    codes = [mpl.path.Path.MOVETO, mpl.path.Path.LINETO, mpl.path.Path.LINETO,
             mpl.path.Path.LINETO, mpl.path.Path.CLOSEPOLY]
    return mpl.path.Path(vertexes, codes)


def _fixed_axes(x_min, x_max, y_min, y_max, figsize: None | tuple[float, float]):
    """Fixes figure size and limits the visible area of the graph to the specified parameters"""
    figsize = ((x_max - x_min) / 2 / 2.54, (y_max - y_min) / 2 / 2.54) if figsize is None else figsize
    plt.figure(figsize=figsize)
    axes = plt.gca()
    axes.set_xlim(x_min, x_max)
    axes.set_ylim(y_min, y_max)
    return axes


def _intersection_with_vertical_line(points: np.ndarray, x0: float) -> float:
    interpolating_func = sc.interpolate.interp1d(*zip(*points))
    return interpolating_func(x0)


if __name__ == '__main__':
    # constant_phase_curve_2signs()
    integration_contour()
    # z_k_position()

    # plt.show()
    plt.savefig('constant_phase_curve', dpi=500)
