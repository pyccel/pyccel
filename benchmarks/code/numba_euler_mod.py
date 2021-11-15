#! /usr/bin/env python3
"""
Functions for solving an ordinary differential equation using Euler's method. The code is adapted from examples written by [J. Burkardt](https://people.sc.fsu.edu/~jburkardt/py_src/py_src.html)
"""
from numba import njit
from numpy import zeros
from numpy import linspace

# ================================================================
@njit(fastmath=True)
def euler (dydt: '()(real, const real[:], real[:])',
           tspan: 'real[:]', y0: 'real[:]', n: int,
           t: 'real[:]', y: 'real[:,:]'):
    """
    Function implementing Euler's method
    """

    t0 = tspan[0]
    t1 = tspan[1]
    dt = ( t1 - t0 ) / float ( n )
    y[0] = y0[:]

    for i in range ( n ):
        dydt ( t[i], y[i,:], y[i+1,:] )
        y[i+1,:] = y[i,:] + dt * y[i+1,:]

# ================================================================
@njit(fastmath=True)
def humps_fun ( x : float ):
    """
    Humps function
    """

    y = 1.0 / ( ( x - 0.3 )**2 + 0.01 ) \
            + 1.0 / ( ( x - 0.9 )**2 + 0.04 ) \
            - 6.0

    return y

# ================================================================
@njit(fastmath=True)
def humps_deriv ( x: 'real', y: 'real[:]', out: 'real[:]' ):
    """
    Derivative of the humps function
    """

    out[0] = - 2.0 * ( x - 0.3 ) / ( ( x - 0.3 )**2 + 0.01 )**2 - 2.0 * ( x - 0.9 ) / ( ( x - 0.9 )**2 + 0.04 )**2

# ================================================================
@njit(fastmath=True)
def euler_humps_test ( tspan: 'real[:]', y0: 'real[:]', n: int ):
    """
    Run n steps of an euler method starting from y0

    Parameters
    ----------
    tspan : array of 2 floats
            The first element is the start time.
            The second element is the end time.
    y0    : array of floats
            The starting point for the evolution
    n     : int
            The number of time steps
    """

    m = len ( y0 )

    t0 = tspan[0]
    t1 = tspan[1]

    t = linspace ( t0, t1, n + 1 )
    y = zeros ( ( n + 1, m ) )

    euler ( humps_deriv, tspan, y0, n, t, y )
