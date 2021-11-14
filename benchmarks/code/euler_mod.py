#! /usr/bin/env python3
from numpy import zeros
from numpy import linspace

# ================================================================
def euler (dydt: '()(real, const real[:], real[:])',
           tspan: 'real[:]', y0: 'real[:]', n: int,
           t: 'real[:]', y: 'real[:,:]'):

    t0 = tspan[0]
    t1 = tspan[1]
    dt = ( t1 - t0 ) / float ( n )
    y[0] = y0[:]

    for i in range ( n ):
        dydt ( t[i], y[i,:], y[i+1,:] )
        y[i+1,:] = y[i,:] + dt * y[i+1,:]

# ================================================================
# pythran export humps_fun(float)
def humps_fun ( x : float ):

        y = 1.0 / ( ( x - 0.3 )**2 + 0.01 ) \
                + 1.0 / ( ( x - 0.9 )**2 + 0.04 ) \
                - 6.0

        return y

# ================================================================
def humps_deriv ( x: 'real', y: 'real[:]', out: 'real[:]' ):

    out[0] = - 2.0 * ( x - 0.3 ) / ( ( x - 0.3 )**2 + 0.01 )**2 - 2.0 * ( x - 0.9 ) / ( ( x - 0.9 )**2 + 0.04 )**2

# ================================================================
# pythran export euler_humps_test(float[:],float[:],int)
def euler_humps_test ( tspan: 'real[:]', y0: 'real[:]', n: int ):


    m = len ( y0 )

    t0 = tspan[0]
    t1 = tspan[1]

    t = linspace ( t0, t1, n + 1 )
    y = zeros ( [ n + 1, m ] )

    euler ( humps_deriv, tspan, y0, n, t, y )
