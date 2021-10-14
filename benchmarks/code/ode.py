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
def midpoint_explicit (dydt: '()(real, const real[:], real[:])',
                       tspan: 'real[:]', y0: 'real[:]', n: int,
                       t: 'real[:]', y: 'real[:,:]'):

    m = len( y0 )
    ym = zeros(m)

    dt = ( tspan[1] - tspan[0] ) / float ( n )

    t[0] = tspan[0]
    y[0,:] = y0[:]

    for i in range ( 0, n ):

        tm = t[i]   + 0.5 * dt
        dydt ( t[i], y[i,:], ym[:] )
        ym[:] = y[i,:] + 0.5 * dt * ym[:]

        t[i+1]   = t[i]   + dt
        dydt ( tm, ym[:], y[i+1,:] )
        y[i+1,:] = y[i,:] + dt * y[i+1,:]

# ================================================================
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
def leapfrog (dydt: '()(real, const real[:], real[:])',
              tspan: 'real[:]', y0: 'real[:]', n: int,
              t: 'real[:]', y: 'real[:,:]'):

    # TODO len(y0) == 2
    m = len( y0 )
    anew = zeros(m)

    t0 = tspan[0]
    tstop = tspan[1]
    dt = ( tstop - t0 ) / n

    for i in range ( 0, n + 1 ):

        if ( i == 0 ):
            t[0]   = t0
            y[0,0] = y0[0]
            y[0,1] = y0[1]
            dydt ( t[i], y[i,:], anew )

        else:
            t[i]   = t[i-1] + dt
            aold   = anew
            y[i,0] = y[i-1,0] + dt * ( y[i-1,1] + 0.5 * dt * aold[1] )
            dydt ( t[i], y[i,:], anew )
            y[i,1] = y[i-1,1] + 0.5 * dt * ( aold[1] + anew[1] )

# ================================================================
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
def predator_prey_deriv ( t: 'real', rf: 'real[:]', out: 'real[:]' ):

    r = rf[0]
    f = rf[1]

    drdt =    2.0 * r - 0.001 * r * f
    dfdt = - 10.0 * f + 0.002 * r * f

    out[0] = drdt
    out[1] = dfdt

# ================================================================
def shm_deriv ( x: 'real', y: 'real[:]', out: 'real[:]' ):

    out[0] =   y[1]
    out[1] = - y[0]

# ================================================================
# pythran export euler_humps_test(float[:],float[:],int)
def euler_humps_test ( tspan: 'real[:]', y0: 'real[:]', n: int ):


    m = len ( y0 )

    t0 = tspan[0]
    t1 = tspan[1]

    t = linspace ( t0, t1, n + 1 )
    y = zeros ( [ n + 1, m ] )

    euler ( humps_deriv, tspan, y0, n, t, y )

# ================================================================
# pythran export midpoint_explicit_humps_test(float[:],float[:],int)
def midpoint_explicit_humps_test ( tspan: 'real[:]', y0: 'real[:]', n: int ):

    m = len ( y0 )

    t0 = tspan[0]
    t1 = tspan[1]

    t = linspace ( t0, t1, n + 1 )
    y = zeros ( [ n + 1, m ] )

    midpoint_explicit ( humps_deriv, tspan, y0, n, t, y )

# ================================================================
# pythran export midpoint_explicit_predator_prey_test(float[:],float[:],int)
def midpoint_explicit_predator_prey_test ( tspan: 'real[:]', y0: 'real[:]', n: int ):

    m = len ( y0 )

    t = zeros ( n + 1 )
    y = zeros ( [ n + 1, m ] )

    midpoint_explicit ( predator_prey_deriv, tspan, y0, n, t, y )

# ================================================================
# pythran export midpoint_fixed_humps_test(float[:],float[:],int)
def midpoint_fixed_humps_test ( tspan: 'real[:]', y0: 'real[:]', n: int ):

    m = len ( y0 )

    t0 = tspan[0]
    t1 = tspan[1]

    t = linspace ( t0, t1, n + 1 )
    y = zeros ( [ n + 1, m ] )

    midpoint_fixed ( humps_deriv, tspan, y0, n, t, y )

# ================================================================
# pythran export midpoint_fixed_predator_prey_test(float[:],float[:],int)
def midpoint_fixed_predator_prey_test ( tspan: 'real[:]', y0: 'real[:]', n: int ):

    m = len ( y0 )

    t = zeros ( n + 1 )
    y = zeros ( [ n + 1, m ] )

    midpoint_fixed ( predator_prey_deriv, tspan, y0, n, t, y )

# ================================================================
def leapfrog_shm_test ( tspan: 'real[:]', y0: 'real[:]', n: int ):

    m = len ( y0 )

    t = zeros ( n + 1 )
    y = zeros ( [ n + 1, m ] )

    leapfrog ( shm_deriv, tspan, y0, n, t, y )

# ================================================================
# pythran export rk4_humps_test(float[:],float[:],int)
def rk4_humps_test ( tspan: 'real[:]', y0: 'real[:]', n: int ):

    m = len ( y0 )

    t = zeros ( n + 1 )
    y = zeros ( [ n + 1, m ] )

    rk4 ( humps_deriv, tspan, y0, n, t, y )

# ================================================================
# pythran export rk4_predator_prey_test(float[:],float[:],int)
def rk4_predator_prey_test ( tspan: 'real[:]', y0: 'real[:]', n: int ):

    m = len ( y0 )

    t = zeros ( n + 1 )
    y = zeros ( [ n + 1, m ] )

    rk4 ( predator_prey_deriv, tspan, y0, n, t, y )
