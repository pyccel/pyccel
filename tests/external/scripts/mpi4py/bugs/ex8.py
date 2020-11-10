# pylint: disable=missing-function-docstring, missing-module-docstring/
from mpi4py import MPI

rank = -1
#we must initialize rank
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


master    = 1
nb_values = 8

block_length = nb_values / size

data = zeros(block_length, 'int')

if rank == master:
    values = zeros(nb_values, 'int')
    for i in range(0, nb_values):
        values[i] = 1000 + i

    print('I, process ', rank ,' send my values array', values)

mpi_scatter (values, block_length, MPI_INTEGER,
             data,   block_length, MPI_INTEGER,
             master, comm, ierr)

print('I, process ', rank, ', received ', data, ' of process ', master)

