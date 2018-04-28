# coding: utf-8

from pyccel.stdlib.internal.mpi import mpi_init
from pyccel.stdlib.internal.mpi import mpi_finalize
from pyccel.stdlib.internal.mpi import mpi_comm_size
from pyccel.stdlib.internal.mpi import mpi_comm_rank
from pyccel.stdlib.internal.mpi import mpi_comm_world
from pyccel.stdlib.internal.mpi import mpi_status_size
from pyccel.stdlib.internal.mpi import mpi_sendrecv
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

if rank == 0:
    partner = 1

if rank == 1:
    partner = 0

msg = rank + 1000
val = -1
tag = 1234
status = zeros(mpi_status_size, 'int')

mpi_sendrecv (msg, 1, MPI_INTEGER, partner, tag,
              val, 1, MPI_INTEGER, partner, tag,
              comm, status, ierr)

print('I, process ', rank, ', I received', val, ' from process ', partner)

mpi_finalize(ierr)
