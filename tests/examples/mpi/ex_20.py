# coding: utf-8

from pyccel.mpi import *

ierr = mpi_init()

comm = mpi_comm_world
size = comm.size
rank = comm.rank

u = range(0, 8)
v = range(0, 8)

uv = tensor(u, v)
x  = zeros(uv, double)
y  = ones(uv, double)

for i,j in uv:
    print ('(i,j) = (', i, j,')')

sync(uv) x

del uv

ierr = mpi_finalize()
