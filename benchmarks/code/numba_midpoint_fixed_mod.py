# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
Functions for solving an ordinary differential equation using the implicit midpoint method for a fixed number of iterations. The code is adapted from examples written by [J. Burkardt](https://people.sc.fsu.edu/~jburkardt/py_src/py_src.html)
To be accelerated with numba
"""
from numba import njit
from numpy import zeros
from numpy import linspace

# ================================================================
@njit(fastmath=True)
def midpoint_fixed (dydt: '()(real, const real[:], real[:])',
                    tspan: 'real[:]', y0: 'real[:]', n: int,
                    t: 'real[:]', y: 'real[:,:]'):
    """
    Function implementing the implicit midpoint method for 10 iterations
    """

    m = len( y0 )
    y1m = zeros(m)
    y2m = zeros(m)

    dt = ( tspan[1] - tspan[0] ) / float ( n )

    it_max = 10
    theta = 0.5

    t[0] = tspan[0]
    y[0,:] = y0

    for i in range ( 0, n ):

        xm = t[i] + theta * dt

        y1m[:] = y[i,:]
        for _ in range ( it_max ):
            dydt ( xm, y1m[:], y2m[:] )
            y1m[:] = y[i,:] + theta * dt * y2m[:]

        t[i+1] = t[i] + dt
        y[i+1,:] = (       1.0 / theta ) * y1m[:] \
                 + ( 1.0 - 1.0 / theta ) * y[i,:]

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
def midpoint_fixed_humps_test ( tspan: 'real[:]', y0: 'real[:]', n: int ):
    """
    Run n steps of an implicit midpoint method with a fixed number of iterations,
    starting from y0

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

    midpoint_fixed ( humps_deriv, tspan, y0, n, t, y )

