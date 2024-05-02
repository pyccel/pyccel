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

mesh = tensor(r_x, r_y)

u = zeros(mesh, double)

#Initialization
x = 0.0
y = 0.0
for i,j in mesh:
    x = i*hx
    y = j*hy

    u[i, j] = x*y*(x-1.0)*(y-1.0)

#Communication
sync(mesh) u

#Delete memory
del mesh

ierr = mpi_finalize()
