# coding: utf-8

from pyccel.stdlib.internal.mpi import mpi_init
from pyccel.stdlib.internal.mpi import mpi_finalize
from pyccel.stdlib.internal.mpi import mpi_comm_size
from pyccel.stdlib.internal.mpi import mpi_comm_rank
from pyccel.stdlib.internal.mpi import mpi_comm_world
from pyccel.stdlib.internal.mpi import mpi_status_size
from pyccel.stdlib.internal.mpi import mpi_scatter
from pyccel.stdlib.internal.mpi import MPI_INTEGER

from numpy import zeros

# we need to declare these variables somehow,
# since we are calling mpi subroutines
ierr = -1
size = -1
rank = -1

mpi_init(ierr)

comm = mpi_comm_world
mpi_comm_size(comm, size, ierr)
mpi_comm_rank(comm, rank, ierr)

master    = 1
nb_values = 8

block_length = nb_values // size

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

mpi_finalize(ierr)
