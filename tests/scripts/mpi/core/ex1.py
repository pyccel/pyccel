# coding: utf-8

from pyccel.stdlib.parallel.mpi import mpi_init
from pyccel.stdlib.parallel.mpi import mpi_finalize
from pyccel.stdlib.parallel.mpi import mpi_comm_size
from pyccel.stdlib.parallel.mpi import mpi_comm_rank
from pyccel.stdlib.parallel.mpi import mpi_comm_world
from pyccel.stdlib.parallel.mpi import mpi_status_size
from pyccel.stdlib.parallel.mpi import mpi_send
from pyccel.stdlib.parallel.mpi import mpi_recv

# we need to declare these variables somehow,
# since we are calling mpi subroutines
ierr = -1
size = -1
rank = -1

mpi_init(ierr)

comm = mpi_comm_world
mpi_comm_size(comm, size, ierr)
mpi_comm_rank(comm, rank, ierr)

n = 4
x = zeros(n, double)
y = zeros((3,2), double)

if rank == 0:
    x = 1.0
    y = 1.0

source = 0
dest   = 1
tagx = 1234
status = zeros(mpi_status_size, int)

if rank == source:
    mpi_send(x, 1, MPI_DOUBLE, dest, tagx, comm, ierr)
    print("processor ", rank, " sent ", x)

if rank == dest:
    mpi_recv(x, 1, MPI_DOUBLE, source, tagx, comm, status, ierr)
    print("processor ", rank, " got  ", x)

mpi_finalize(ierr)
