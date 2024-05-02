# coding: utf-8

from pyccel.mpi import *

ierr = mpi_init()

comm     = mpi_comm_world
nb_procs = comm.size
rank     = comm.rank

root      = 1
nb_values = 8

block_length = nb_values / nb_procs

data   = zeros(block_length, double)

if rank == root:
    values = zeros(nb_values, double)
    for i in range(0, nb_values):
        values[i] = 1000 + i
    print(('I, process ', rank ,' send my values array', values))

ierr = comm.scatter(values, data, root)
print(('I, process ', rank, ', received ', data, ' of process ', root))

ierr = mpi_finalize()
