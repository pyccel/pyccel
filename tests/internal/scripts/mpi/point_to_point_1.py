# coding: utf-8

from pyccel.stdlib.internal.mpi import mpi_init
from pyccel.stdlib.internal.mpi import mpi_finalize
from pyccel.stdlib.internal.mpi import mpi_comm_size
from pyccel.stdlib.internal.mpi import mpi_comm_rank
from pyccel.stdlib.internal.mpi import mpi_comm_world
from pyccel.stdlib.internal.mpi import mpi_status_size
from pyccel.stdlib.internal.mpi import mpi_send
from pyccel.stdlib.internal.mpi import mpi_recv
from pyccel.stdlib.internal.mpi import MPI_DOUBLE

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

nx = 4
x = zeros(nx)

if rank == 0:
    x[:] = 1.0

source = 0
dest   = 1

# ...
tag1 = 1234
if rank == source:
    x[1] = 2.0
    mpi_send(x[1], 1, MPI_DOUBLE, dest, tag1, comm, ierr)
    print("> processor ", rank, " sent x(1) = ", x)
# ...

mpi_finalize(ierr)
