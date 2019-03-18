#$ header metavar print=True

from pyccel.stdlib.internal.fitpack import bispev



#$ header function bispev2(double[:] , int, double[:], int, double[:], int, int, double[:], int, double[:], int, double[:,:], int)
def bispev2(tx, nx, ty, ny, c, kx, ky, x, mx, y, my, z, ierr):

    from numpy import empty
    lwrk = mx*(kx+1)+my*(ky+1)
    wrk  = empty(lwrk)
    kwrk = mx+my
    iwrk = empty(kwrk,'int')
    bispev(tx, nx, ty, ny, c, kx, ky, x, mx, y, my, z, wrk, lwrk, iwrk, kwrk, ierr)



#$ header macro (z,ierr), _bispev(tx, ty, c, kx, ky, x, y) := bispev2(tx, tx.count , ty, ty.count, c, kx, ky, x, x.count, y, y.count, z, ierr)
