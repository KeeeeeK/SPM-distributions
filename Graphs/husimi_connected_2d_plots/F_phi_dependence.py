import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from SPM_distributions.Husimi.F_normalized import Fn_2b


def Fn_plot():
    """
    Draws F normalized depending on angle of its parameter.
    Simply speaking |F(r*exp(1j*phi))|exp(-r) as function on phi.
    """
    fig, ax = plt.subplots(layout=None)
    n_dots = 10 ** 4
    phi = np.linspace(0, 2 * np.pi, n_dots)
    set_label(ax, (r'$\phi$', [1.04, 0]),
                  (r'$\left|F(r e^{i\phi}, e^{i\Gamma})\right|e^{-r}$', [-0.01, 1.03]))
    _pi_x_axis(ax)
    plt.plot([], [], ' ', label="$\Gamma = 10^{-4}$", c='C10')
    for r, gamma in ((10 ** 4, 10 ** -4), (2 * 10 ** 4, 10 ** -4), (3.1 * 10 ** 4, 10 ** -4))[::-1]:
        vals = Fn_2b(r, phi, gamma)
        plt.plot(phi, vals, label=f'r={r * gamma}/$\Gamma$')
    plt.legend()


def set_label(axes, x_label_params: tuple[str, list[float]], y_label_params):
    label_prop = dict(rotation=0)
    for set_label_ax, axis, label, label_coords in (
            (axes.set_xlabel, axes.xaxis, *x_label_params),
            (axes.set_ylabel, axes.yaxis, *y_label_params)):
        set_label_ax(label, label_prop)
        axis.set_label_coords(*label_coords)


def _pi_x_axis(axes):
    """
    This function fixes x axis from 0 to 2pi and displays ticks in multiples of pi
    """
    axes.axhline(0, color='black', lw=2)
    axes.set_xlim(0, 2 * np.pi)
    axes.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 6))
    axes.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 24))
    axes.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter(denominator=6, number=np.pi, latex='\pi')))


def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    # thanks to https://stackoverflow.com/questions/40642061/how-to-set-axis-ticks-in-multiples-of-pi-python-matplotlib
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int_(np.rint(den * x / number))
        com = gcd(num, den)
        (num, den) = (int(num / com), int(den / com))
        if den == 1:
            if num == 0:
                return r'$0$'
            if num == 1:
                return r'$%s$' % latex
            elif num == -1:
                return r'$-%s$' % latex
            else:
                return r'$%s%s$' % (num, latex)
        else:
            if num == 1:
                return r'$\frac{%s}{%s}$' % (latex, den)
            elif num == -1:
                return r'$\frac{-%s}{%s}$' % (latex, den)
            else:
                return r'$\frac{%s%s}{%s}$' % (num, latex, den)

    return _multiple_formatter


if __name__ == '__main__':
    Fn_plot()
    # plt.show()
    plt.savefig('Fn_depending_on_phi', dpi=600)
