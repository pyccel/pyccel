""" A test where the function integrated over a grid is varied. The integration is translated without rewriting the original function.
"""
#pylint: disable=wrong-import-position, unnecessary-lambda-assignment
from numpy import linspace, exp

# INTEGRATE_GRID

def midpoint_rule(nx : int, ny : int, x_start : float, x_end : float, y_start : float, y_end : float):
    """
    Integrate the function test_func using the midpoint rule.
    """
    dx = (x_end - x_start) / nx
    dy = (y_end - y_start) / ny
    xs = linspace(x_start + 0.5*dx, x_end - 0.5*dx, nx)
    ys = linspace(y_start + 0.5*dy, y_end - 0.5*dy, ny)

    result = 0.0
    for i in range(nx):
        for j in range(ny):
            # result += exp(-(xs[i]**2 + ys[j]**2)) * dx * dy
            # result += exp(-(xs[i]**3 + ys[j]**2)) * dx * dy
            # result += exp(-(xs[i]**2 + ys[j]**3)) * dx * dy
            result += test_func(xs[i], ys[j]) * dx * dy
    return result

# END_INTEGRATE_GRID

# COMPILE

from pyccel import epyccel

test_func = lambda x, y : exp(-(x**2 + y**2))
compiled_integrator = epyccel(midpoint_rule)

# END_COMPILE

# TEST

# area = midpoint_rule(1000, 1000, -5., 5., -5., 5.)
area = compiled_integrator(1000, 1000, -5., 5., -5., 5.)
print(area)

# END_TEST

# MULTIPLE_TESTS

test_func = lambda x, y : exp(-(x**2 + y**2))
integrate_test_1 = epyccel(midpoint_rule)
test_func = lambda x, y : exp(-(x**2 + y**2))
integrate_test_2 = epyccel(midpoint_rule)

# END_MULTIPLE_TESTS
