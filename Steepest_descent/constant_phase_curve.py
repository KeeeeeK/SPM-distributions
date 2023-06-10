from typing import Callable

import scipy as sc
import sympy as sp
import numpy as np

complex_func_type = Callable[[complex], complex] | Callable[[np.ndarray], np.ndarray]


def constant_phase_curve(z: sp.Symbol,
                         analytic_func: sp.Expr,
                         start_point: tuple[float, float],
                         steps_params=tuple[float, int, int]) -> np.ndarray:
    """
    Gives the coordinates of points along the constant phase curve of the function exp(analytic_func)

    :param z: the argument of the following analytic function
    :param analytic_func: the analytic function of z
    :param start_point: x0, y0
    The saddle-point from which the curve will be drawn.
    That is exactly at this point func'(x0+1j*y0)==0 AND func''(x0+1j*y0)!=0.
    Without this the function will not work!
    It is implied that the constant phase curve will not pass through any other saddle-points.
    :param steps_params: tuple of step, steps_backward, steps_forward
    :var step is the approximate step size along the curve. It is required to make it relatively small.
    :var steps_backward, steps_forward: number of steps forward and backward along the desired curve.
    The "forward" direction is the direction that corresponds to the positive projection on the Re(z) axis
    (and also Re(f(z)) declines, Im(f(z)) a constant)

    :return: An array of constant phase curve coordinate pairs.
    """
    x0, y0 = start_point
    step, steps_backward, steps_forward = steps_params

    derivative = sp.diff(analytic_func)
    cos_chi, sin_chi = _positive_direction(z, derivative, (x0, y0))

    num_func, num_derivative = sp.lambdify(z, analytic_func), sp.lambdify(z, derivative)
    forward_points = _step_algorithm(num_func, num_derivative, (x0, y0),
                                     (+cos_chi, +sin_chi), step, steps_forward)
    backward_points = _step_algorithm(num_func, num_derivative, (x0, y0),
                                      (-cos_chi, -sin_chi), step, steps_backward)
    return np.concatenate((backward_points[::-1], ((x0, y0),), forward_points))


def _positive_direction(z: sp.Symbol,
                        derivative: sp.Expr,
                        start_point: tuple[float, float],
                        print_derivative: bool = False) -> tuple[float, float]:
    """
    Positive direction = the direction along which:
    1. Corresponds to the positive projection on the Re(z) axis
    2. Re(f(z)) decreases
    3. Im(f(z)) constant

    :param z: the argument of the analytic function
    :param derivative: the derivative of analytic function of z
    :param start_point: coordinates of starting point
    :param print_derivative: if True, prints the value of derivative in start point
    :return the direction called "forward" in constant_phase_curve description in format (cos(chi), sin(chi)),
    where 'chi' is an angle between "forward" direction and Re(z)
    """
    x0, y0 = start_point
    z0: complex = x0 + 1j * y0
    if print_derivative is True:
        print(f'derivative in start point = {complex(derivative.evalf(subs={z: z0}))}')
    if not np.isclose(np.complex_(derivative.evalf(subs={z: z0})), 0):
        raise NonZeroFPrime('The algorithm expects a zero first derivative at the saddle point')
    second_derivative = np.complex_(sp.diff(derivative, z).evalf(subs={z: z0}))
    if np.isclose(second_derivative, 0):
        raise ZeroFPrimePrime('The algorithm expects a nonzero second derivative at the saddle point')
    cos_2chi = - np.real(second_derivative) / np.abs(second_derivative)
    sin_chi_sign = np.sign(np.imag(second_derivative)) if np.imag(second_derivative) != 0 else 1
    return np.sqrt(1 / 2 * (1 + cos_2chi)), np.sqrt(1 / 2 * (1 - cos_2chi)) * sin_chi_sign


def _step_algorithm(num_func: complex_func_type,
                    num_derivative: complex_func_type,
                    initial_point: tuple[float, float],
                    initial_direction: tuple[float, float],
                    step: float,
                    num_steps: int) -> np.ndarray:
    """
    :param num_func: complex -> complex, function f
    :param num_derivative: complex -> complex, f prime
    :param initial_point: a pair of initial point coordinates
    :param initial_direction: direction in which to look for the next point of the constant phase curve.
    Parameter is given by the pair of cosine and sine of the angle between the direction and Re(z)
    :param step: approximate step size along the curve. It is required to make it relatively small.
    :param num_steps: the number of steps walked in the specified direction.
    The approximate length of the resulting curve will be step*num_steps
    :return: points in the (x, y) format on the constant phase curve of the function exp(f(z))
    """
    x0, y0 = initial_point
    phase = np.imag(num_func(x0 + 1j * y0))
    # Once at a point lying on the curve of the constant phase, it is obvious that it is necessary to move in the
    # direction of the steepest descent, for which the calculation of the derivative at this point is necessary. To more
    # accurately account for deviations of the second order of smallness, the function _exact_next_z is used
    filling_arr = np.empty((num_steps, 2))
    current_point, current_direction = initial_point, initial_direction
    for i in range(num_steps):
        current_point, current_direction = \
            _next_point_and_direction(num_func, num_derivative, current_point, current_direction, step, phase)
        filling_arr[i][0], filling_arr[i][1] = current_point
    return filling_arr


def _next_point_and_direction(num_func: complex_func_type,
                              num_derivative: complex_func_type,
                              current_point: tuple[float, float],
                              current_direction: tuple[float, float],
                              step: float,
                              phase: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    :param num_func, num_derivative: same as in _step_algorithm
    :param current_point, current_direction: point on the constant phase curve and direction of tangent to the curve.
    The format is the same as in _step_algorithm
    :param step: same as in _step_algorithm
    :param phase: the imaginary part of the function that persists along the entire curve of the constant phase
    :return: tuple of point and direction in the same format as in _step_algorithm
    """
    cos_chi, sin_chi = current_direction

    z_exact = _exact_next_z(num_func, current_point, current_direction, step, phase)
    f_prime = num_derivative(z_exact)
    mb_cos_chi, mb_sin_chi = np.real(f_prime) / np.abs(f_prime), -np.imag(f_prime) / np.abs(f_prime)
    if np.abs(mb_cos_chi - cos_chi) + np.abs(mb_sin_chi - sin_chi) > \
            np.abs(mb_cos_chi + cos_chi) + np.abs(mb_sin_chi + sin_chi):
        mb_cos_chi, mb_sin_chi = -mb_cos_chi, -mb_sin_chi

    return (np.real(z_exact), np.imag(z_exact)), (mb_cos_chi, mb_sin_chi)


def _exact_next_z(num_func: complex_func_type,
                  current_point: tuple[float, float],
                  current_direction: tuple[float, float],
                  step: float,
                  phase: float) -> np.complex_:
    """Params are the same as in _next_point_and_direction"""
    x0, y0 = current_point
    cos_chi, sin_chi = current_direction

    # Expected coordinates of the next point
    # (The actual position of the point on the constant width curve may vary)
    x_es, y_es = x0 + step * cos_chi, y0 + step * sin_chi
    # A naturally parameterized straight line passing through the expected point and perpendicular to the
    # estimated direction
    z_of_t = lambda t: x_es + sin_chi * t + 1j * (y_es - cos_chi * t)
    # the amount of deviation from the tangent direction
    t_sol = sc.optimize.root_scalar(lambda t: np.imag(num_func(z_of_t(t))) - phase, x0=-step, x1=step)
    return np.complex_(z_of_t(t_sol.root))


class NonZeroFPrime(Exception):
    pass


class ZeroFPrimePrime(Exception):
    pass
