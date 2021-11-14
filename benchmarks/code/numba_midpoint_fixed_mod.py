#! /usr/bin/env python3
from numba import njit
from numpy import zeros
from numpy import linspace

# ================================================================
@njit(fastmath=True)
def midpoint_fixed (dydt: '()(real, const real[:], real[:])',
                    tspan: 'real[:]', y0: 'real[:]', n: int,
                    t: 'real[:]', y: 'real[:,:]'):

    m = len( y0 )
    y1m = zeros(m)
    y2m = zeros(m)

    dt = ( tspan[1] - tspan[0] ) / float ( n )

    it_max = 10
    theta = 0.5

    t[0] = tspan[0];
    y[0,:] = y0

    for i in range ( 0, n ):

        xm = t[i] + theta * dt

        y1m[:] = y[i,:]
        for j in range ( 0, it_max ):
            dydt ( xm, y1m[:], y2m[:] )
            y1m[:] = y[i,:] + theta * dt * y2m[:]

        t[i+1] = t[i] + dt
        y[i+1,:] = (       1.0 / theta ) * y1m[:] \
                 + ( 1.0 - 1.0 / theta ) * y[i,:]

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
def midpoint_fixed_humps_test ( tspan: 'real[:]', y0: 'real[:]', n: int ):

    m = len ( y0 )

    t0 = tspan[0]
    t1 = tspan[1]

    t = linspace ( t0, t1, n + 1 )
    y = zeros ( ( n + 1, m ) )

    midpoint_fixed ( humps_deriv, tspan, y0, n, t, y )

