# coding: utf-8

from pyccel.stdlib.internal.mpi import mpi_init
from pyccel.stdlib.internal.mpi import mpi_finalize
from pyccel.stdlib.internal.mpi import mpi_comm_size
from pyccel.stdlib.internal.mpi import mpi_comm_rank
from pyccel.stdlib.internal.mpi import mpi_comm_world
from pyccel.stdlib.internal.mpi import mpi_status_size
from pyccel.stdlib.internal.mpi import mpi_isend
from pyccel.stdlib.internal.mpi import mpi_irecv
from pyccel.stdlib.internal.mpi import mpi_waitall
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

n = 4
x = zeros(n)
y = zeros(n)

if rank == 0:
    x[:] = 1.0
    y[:] = 2.0

# ...
tag0 = 1234
tag1 = 5678

reqs = zeros(4, 'int')
# ...

# ...
prev = rank - 1
next = rank + 1
if rank == 0:
    prev = size - 1
if rank == size - 1:
    next = 0
# ...

# ...
mpi_irecv(x, n, MPI_DOUBLE, prev, tag0, comm, reqs[0], ierr)
mpi_irecv(y, n, MPI_DOUBLE, next, tag1, comm, reqs[1], ierr)

mpi_isend(x, n, MPI_DOUBLE, prev, tag1, comm, reqs[2], ierr)
mpi_isend(y, n, MPI_DOUBLE, next, tag0, comm, reqs[3], ierr)
# ...

# ...
statuses = zeros((mpi_status_size, n), 'int')
mpi_waitall(n, reqs, statuses, ierr)
# ...

mpi_finalize(ierr)
