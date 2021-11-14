#! /usr/bin/env python3
from numba import njit
from numpy import zeros
from numpy import linspace

# ================================================================
@njit(fastmath=True)
def rk4 (dydt: '()(real, const real[:], real[:])',
         tspan: 'real[:]', y0: 'real[:]', n: int,
         t: 'real[:]', y: 'real[:,:]'):

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
@njit(fastmath=True)
def humps_fun ( x : float ):

        y = 1.0 / ( ( x - 0.3 )**2 + 0.01 ) \
                + 1.0 / ( ( x - 0.9 )**2 + 0.04 ) \
                - 6.0

        return y

# ================================================================
@njit(fastmath=True)
def humps_deriv ( x: 'real', y: 'real[:]', out: 'real[:]' ):

    out[0] = - 2.0 * ( x - 0.3 ) / ( ( x - 0.3 )**2 + 0.01 )**2 - 2.0 * ( x - 0.9 ) / ( ( x - 0.9 )**2 + 0.04 )**2

# ================================================================
@njit(fastmath=True)
def rk4_humps_test ( tspan: 'real[:]', y0: 'real[:]', n: int ):

    m = len ( y0 )

    t = zeros ( n + 1 )
    y = zeros ( [ n + 1, m ] )

    rk4 ( humps_deriv, tspan, y0, n, t, y )

