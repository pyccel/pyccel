# coding: utf-8
from sympy import cos, symbols, pi, simplify, Symbol
from sympy.calculus import finite_diff_weights

import numpy as np

__all__ = ['compute_stencil_uniform']

def compute_stencil_uniform(order, n, x_value, h_value, x0=0.):
    """
    computes a stencil of Order order

    order: int
        derivative order
    n: int
        number of points - 1
    x_value: float
        value of the grid point
    h_value: float
        mesh size
    x0: float
        real number around which we compute the Taylor expansion.
    """
    h,x = symbols('h x')
    if n % 2 == 0:
        m = n / 2
        xs = [x+h*i for i in range(-m,m+1)]
    else:
        m = n / 2
        xs = [x+h*i+h/2 for i in range(-m-1,m+1)]

    cs = finite_diff_weights(order, xs, x0)[order][n]
    cs = [simplify(c) for c in cs]

    cs = [simplify(expr.subs(h, h_value)) for expr in cs]
    xs = [simplify(expr.subs(h, h_value)) for expr in xs]

    cs = [simplify(expr.subs(x, x_value)) for expr in cs]
    xs = [simplify(expr.subs(x, x_value)) for expr in xs]

    return np.array(cs, dtype=float)

def compute_stencil(order, n, x_value, h_value, x0=0.):
    """
    computes a stencil of Order order
    """
    h,x = symbols('h x')
    xs = [x+h*cos(i*pi/n) for i in range(n,-1,-1)]
    cs = finite_diff_weights(order, xs, x0)[order][n]
    cs = [simplify(c) for c in cs]

    cs = [simplify(expr.subs(h, h_value)) for expr in cs]
    xs = [simplify(expr.subs(h, h_value)) for expr in xs]

    cs = [simplify(expr.subs(x, x_value)) for expr in cs]
    xs = [simplify(expr.subs(x, x_value)) for expr in xs]

    return xs, cs

##############################################
if __name__ == "__main__":
#    xs, cs = compute_stencil(1, 4, 0.5, 0.25)
    cs = compute_stencil_uniform(2, 4, 0., 0.25)
    print(cs)
    cs = compute_stencil_uniform(2, 3, 0., 0.25)
    print(cs)

