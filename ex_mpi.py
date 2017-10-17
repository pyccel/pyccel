# coding: utf-8

from pyccel.mpi import *

ierr = mpi_init()

ntx = 8
nty = 8
r_x = range(0, ntx)
r_y = range(0, nty)

#Grid spacing
hx = 1.0/(ntx+1)
hy = 1.0/(nty+1)

mesh = MPI_Tensor_NEW(r_x, r_y)

ierr = mpi_finalize()
