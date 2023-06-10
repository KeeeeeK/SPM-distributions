import numpy as np
import numpy.typing as npt
import sympy as sp

from constant_phase_curve import complex_func_type


def numeric_re_along_curve(num_func: complex_func_type, points: npt.NDArray[tuple[float, float]]) -> npt.NDArray[float]:
    """
    :param num_func: complex -> complex
    :param points: array of coordinates of points
    :return: real(num_func(x+1j*y)) values at given points
    """
    return np.array(tuple(map(lambda point: np.real(num_func(point[0] + 1j * point[1])), points)))


def analytic_re_along_curve(z: sp.Symbol,
                            analytic_func: sp.Expr,
                            points: npt.NDArray[tuple[float, float]]) -> npt.NDArray[float]:
    """The same as numeric_re_along_curve but instead of num_func uses its form in sympy"""
    num_func = sp.lambdify(z, analytic_func)
    return numeric_re_along_curve(num_func, points)
