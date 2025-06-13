#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
Module exposing the fitpack library function to pyccel (see  http://www.netlib.org/dierckx/).
"""
#$ header metavar print=True
import numpy as np
from pyccel.stdlib.internal.fitpack import bispev



def bispev2(tx : 'float64[:]', nx : 'int', ty : 'float64[:]', ny : int,
            c : 'float64[:]', kx : int, ky : int, x : 'float64[:]',
            mx : int, y : 'float64[:]', my : int, z : 'float64[:]', ierr : int):

def bispev(tx : 'float64[:]', ty : 'float64[:]', c : 'float64[:]',
           kx : int, ky : int, x : 'float64[:]', y : 'float64[:]'):
    nx = tx.size
    ny = ty.size
    mx = x.size
    my = y.size
    z = np.empty(mx*my)
    lwrk = mx*(kx+1)+my*(ky+1)
    work  = np.empty(lwrk)
    kwrk = mx+my
    iwrk = np.empty(kwrk, dtype=np.int32)
    bispev(tx, nx, ty, ny, c, kx, ky, x, mx, y, my, z, work, lwrk, iwrk, kwrk, ierr)

    return z, ierr



