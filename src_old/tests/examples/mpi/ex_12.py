# coding: utf-8

from pyccel.mpi import *

ierr = mpi_init()

comm = mpi_comm_world
size = comm.size
rank = comm.rank

if rank == 0:
    value = 1000
else:
    value = rank

sum_value = 0

ierr = comm.allreduce (value, sum_value, '+')

print(('I, process ', rank, ', have the global sum value ', sum_value))

ierr = mpi_finalize()
