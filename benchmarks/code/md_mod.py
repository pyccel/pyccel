# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
Functions for running a small molecular dynamics simulation. The code is adapted from examples written by [J. Burkardt](https://people.sc.fsu.edu/~jburkardt/py_src/py_src.html)
To be accelerated with pyccel or pythran
"""
from numpy import zeros
from numpy import sqrt
from numpy import pi
from numpy import sin

# ================================================================
def compute ( p_num: int, d_num: int, pos: 'double[:,:]', vel: 'double[:,:]',
             mass: float, force: 'double[:,:]' ):
    """ Calculate the energy and forces associated with the current configuration
    """

    rij = zeros ( d_num )

    potential = 0.0

    for i in range ( 0, p_num ):
        #
        #  Compute the potential energy and forces.
        #
        for j in range ( 0, p_num ):
            if ( i != j ):
                #  Compute RIJ, the displacement vector.
                for k in range ( 0, d_num ):
                    rij[k] = pos[k,i] - pos[k,j]

                #  Compute D and D2, a distance and a truncated distance.
                d = 0.0
                for k in range ( 0, d_num ):
                    d = d + rij[k] ** 2

                d = sqrt ( d )
                # TODO BUG
#                d2 = min ( d, pi / 2.0 )
                if d < pi/2.0:
                    d2 = d
                else:
                    d2 = pi/2.0

                #  Attribute half of the total potential energy to particle J.
                potential = potential + 0.5 * sin ( d2 ) * sin ( d2 )

                #  Add particle J's contribution to the force on particle I.
                for k in range ( 0, d_num ):
                    force[k,i] = force[k,i] - rij[k] * sin ( 2.0 * d2 ) / d
    #
    #  Compute the kinetic energy.
    #
    kinetic = 0.0
    for k in range ( 0, d_num ):
        for j in range ( 0, p_num ):
            kinetic = kinetic + vel[k,j] ** 2

    kinetic = 0.5 * mass * kinetic

    return potential, kinetic

# ================================================================
def update ( p_num: int, d_num: int, pos: 'double[:,:]', vel: 'double[:,:]',
            force: 'double[:,:]', acc: 'double[:,:]', mass: float, dt: float ):
    """ Update the position, velocity and force of the particles
    """

    rmass = 1.0 / mass
    #
    #  Update positions.
    #
    pos += vel * dt + 0.5 * acc * dt * dt
    #
    #  Update velocities.
    #
    vel += 0.5 * dt * ( force * rmass + acc )
    #
    #  Update accelerations.
    #
    acc[:] = force * rmass

# ================================================================
def r8mat_uniform_ab ( r: 'double[:,:]', m: int, n: int, a: float, b: float, seed: int ):
    """ Fill r with random numbers with a uniform distribution
    """

    i4_huge = 2147483647

    if ( seed < 0 ):
        seed = seed + i4_huge

    elif ( seed == 0 ):
#        print ( '' ) # TODO error in Pyccel
        print ( 'R8MAT_UNIFORM_AB - Fatal error!' )
        print ( '  Input SEED = 0!' )
        # TODO must stop here
#        sys.exit ( 'R8MAT_UNIFORM_AB - Fatal error!' )

    elif ( seed > 0 ):

        for j in range ( 0, n ):
            for i in range ( 0, m ):

                k = ( seed // 127773 )

                seed = 16807 * ( seed - k * 127773 ) - k * 2836

                seed = ( seed % i4_huge )

                if ( seed < 0 ):
                    seed = seed + i4_huge

                r[i,j] = a + ( b - a ) * seed * 4.656612875E-10

    return seed

# ================================================================
def initialize ( pos: 'double[:,:]', p_num: int, d_num: int ):
    """ Initialise the positions of the particles
    """
    #  Positions.
    seed = 123456789
    seed = r8mat_uniform_ab ( pos, d_num, p_num, 0.0, 10.0, seed )

# ================================================================
def md (d_num: int, p_num: int, step_num: int, dt: float,
        vel: 'double[:,:]', acc: 'double[:,:]',
        force: 'double[:,:]', pos: 'double[:,:]'):
    """ Run molecular dynamics simulation
    """

    mass = 1.0

    initialize ( pos, p_num, d_num )
    compute ( p_num, d_num, pos, vel, mass, force )

    for _ in range ( step_num ):
        update ( p_num, d_num, pos, vel, force, acc, mass, dt )
        compute ( p_num, d_num, pos, vel, mass, force )

# ================================================================
# pythran export test_md(int,int,int,float)
def test_md ( d_num : int = 3, p_num : int = 100, step_num : int = 10, dt : float = 0.1 ):
    """ Run molecular dynamics test
    """

    #  Velocities.
    vel = zeros ( ( d_num, p_num ) )
    #  Accelerations.
    acc = zeros ( ( d_num, p_num ) )
    # Forces
    force = zeros ( ( d_num, p_num ) )
    # Positions
    pos = zeros ( ( d_num, p_num ) )

    md(d_num, p_num, step_num, dt, vel, acc, force, pos)
