# coding: utf-8

from pyccel.mpi import *

ierr = mpi_init()

comm     = mpi_comm_world
nb_procs = comm.size
rank     = comm.rank

nb_values = 8

block_length = nb_values / nb_procs

values = zeros(nb_values, double)
for i in range(0, nb_values):
    values[i] = 1000 + rank*nb_values + i
print(('I, process ', rank, 'sent my values array : ', values))

data = zeros(nb_values, double)
ierr = comm.alltoall(values, data, block_length)

print(('I, process ', rank, ', received ', data))

ierr = mpi_finalize()


