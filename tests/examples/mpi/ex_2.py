# coding: utf-8

from pyccel.mpi import *

ierr = mpi_init()

comm = mpi_comm_world
size = comm.size
rank = comm.rank

n = 4
x = zeros(n, double)

if rank == 0:
    x = 1.0

source = 0
dest   = 1

tagt = 5678
if rank == source:
    x[1] = 2.0
    ierr = comm.send(x[1], dest, tagt)
    print(("processor ", rank, " sent x(1) = ", x[1]))

ierr = mpi_finalize()
