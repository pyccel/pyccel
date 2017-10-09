# coding: utf-8

from pyccel.mpi import *

ierr = mpi_init()

comm = mpi_comm_world
size = comm.size
rank = comm.rank

u = range(0, 8)
v = range(0, 8)

uv = tensor(u, v)

x = 0
for i,j in uv:
    x = x + 1

    print ('(i,j) = (', i, j,')')

del uv

ierr = mpi_finalize()
