# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
Functions for solving a non-linear convection equation. The code is adapted from examples written by [J. Burkardt](https://people.sc.fsu.edu/~jburkardt/py_src/py_src.html)
To be accelerated with numba
"""

from numba import njit

@njit(fastmath=True)
def nonlinearconv_1d(u: 'float[:]', un: 'float[:]',
                     nt: int, nx: int, dt: float, dx: float):
    """ Solve a non-linear convection equation
    """

    for _ in range(nt):
        un[:] = u[:]
        for i in range(1, nx):
            u[i] = un[i] - un[i] * dt / dx * (un[i] - un[i-1])

