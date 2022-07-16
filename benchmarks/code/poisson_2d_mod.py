# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
Functions for solving a Poisson equation. The code is adapted from examples written by [J. Burkardt](https://people.sc.fsu.edu/~jburkardt/py_src/py_src.html)
To be accelerated with pyccel or pythran
"""

# pythran export poisson_2d(float[:,:], float[:,:], float[:,:], int, int, int, float, float)
def poisson_2d(p: 'float[:,:]', pd: 'float[:,:]', b: 'float[:,:]',
               nx: int, ny: int, nt: int, dx: float, dy: float):
    """ Solve the 2D poisson equation
    """

    row, col = p.shape
    # Source
    b[ny // 4, nx // 4]  = 100
    b[3 * ny // 4, 3 * nx // 4] = -100


    for _ in range(nt):
        pd[:,:] = p[:,:]

        for j in range(2, row):
            for i in range(2, col):
                p[j-1, i-1] = (((pd[j-1, i] + pd[j-1, i-2]) * dy**2 +
                                (pd[j, i-1] + pd[j-2, i-1]) * dx**2 -
                                b[j-1, i-1] * dx**2 * dy**2) /
                                (2 * (dx**2 + dy**2)))
        p[0, :] = 0
        p[ny-1, :] = 0
        p[:, 0] = 0
        p[:, nx-1] = 0

