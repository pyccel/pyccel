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


npts    = (4,32)
periods = (False,False)
reorder = False
pads    = (1,1)

mesh = MPI_Tensor(npts, periods, reorder, pads)

x = 0.0
y = 0.0
for i,j in mesh:
    x = i*hx
    y = j*hy

del mesh

ierr = mpi_finalize()
