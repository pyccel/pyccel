# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
Functions for solving an ordinary differential equation using the fourth order Runge-Kutta method. The code is adapted from examples written by [J. Burkardt](https://people.sc.fsu.edu/~jburkardt/py_src/py_src.html)
To be accelerated with pyccel or pythran
"""
from numpy import zeros

# ================================================================
def rk4 (dydt: '()(real, const real[:], real[:])',
         tspan: 'real[:]', y0: 'real[:]', n: int,
         t: 'real[:]', y: 'real[:,:]'):
    """
    Function implementing a fourth order Runge-Kutta method
    """

    m = len( y0 )
    f1 = zeros(m)
    f2 = zeros(m)
    f3 = zeros(m)
    f4 = zeros(m)

    tfirst = tspan[0]
    tlast = tspan[1]
    dt = ( tlast - tfirst ) / n

    t[0] = tspan[0]
    y[0,:] = y0[:]

    for i in range ( 0, n ):

        dydt ( t[i],            y[i,:], f1[:] )
        dydt ( t[i] + dt / 2.0, y[i,:] + dt * f1[:] / 2.0, f2[:] )
        dydt ( t[i] + dt / 2.0, y[i,:] + dt * f2[:] / 2.0, f3[:] )
        dydt ( t[i] + dt,       y[i,:] + dt * f3[:], f4[:] )

        t[i+1] = t[i] + dt
        y[i+1,:] = y[i,:] + dt * ( f1[:] + 2.0 * f2[:] + 2.0 * f3[:] + f4[:] ) / 6.0

# ================================================================
# pythran export humps_fun(float)
def humps_fun ( x : float ):
    """
    Humps function
    """

    y = 1.0 / ( ( x - 0.3 )**2 + 0.01 ) \
            + 1.0 / ( ( x - 0.9 )**2 + 0.04 ) \
            - 6.0

    return y

# ================================================================
def humps_deriv ( x: 'real', y: 'real[:]', out: 'real[:]' ):
    """
    Derivative of the humps function
    """

    out[0] = - 2.0 * ( x - 0.3 ) / ( ( x - 0.3 )**2 + 0.01 )**2 - 2.0 * ( x - 0.9 ) / ( ( x - 0.9 )**2 + 0.04 )**2

# ================================================================
# pythran export rk4_humps_test(float[:],float[:],int)
def rk4_humps_test ( tspan: 'real[:]', y0: 'real[:]', n: int ):
    """
    Run n steps of a fourth-order Runge-Kutta method starting from y0

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

    t = zeros ( n + 1 )
    y = zeros ( [ n + 1, m ] )

    rk4 ( humps_deriv, tspan, y0, n, t, y )

