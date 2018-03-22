# coding: utf-8

from pyccel.mpi import *

ierr = mpi_init()

comm     = mpi_comm_world
nb_procs = comm.size
rank     = comm.rank

root      = 1
nb_values = 8

block_length = nb_values / nb_procs

values = zeros(block_length, double)
for i in range(0, block_length):
    values[i] = 1000 + rank*nb_values + i
print(('I, process ', rank, 'sent my values array : ', values))

data = zeros(nb_values, double)
ierr = comm.gather(values, data, root)

if rank == root:
    print(('I, process ', rank, ', received ', data, ' of process ', root))

ierr = mpi_finalize()
