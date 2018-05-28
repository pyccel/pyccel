# coding: utf-8

from pyccel.stdlib.internal.mpi import mpi_init
from pyccel.stdlib.internal.mpi import mpi_finalize
from pyccel.stdlib.internal.mpi import mpi_comm_size
from pyccel.stdlib.internal.mpi import mpi_comm_rank
from pyccel.stdlib.internal.mpi import mpi_comm_world
from pyccel.stdlib.internal.mpi import mpi_status_size
from pyccel.stdlib.internal.mpi import mpi_bcast
from pyccel.stdlib.internal.mpi import MPI_INTEGER

# we need to declare these variables somehow,
# since we are calling mpi subroutines
ierr = -1
size = -1
rank = -1

mpi_init(ierr)

comm = mpi_comm_world
mpi_comm_size(comm, size, ierr)
mpi_comm_rank(comm, rank, ierr)

master = 1
if rank == master:
    msg = rank + 1000
else:
    msg = 0

mpi_bcast (msg, 1, MPI_INTEGER, master, comm, ierr)

print('I, process ', rank, ', received ', msg, ' from process ', master)

mpi_finalize(ierr)
