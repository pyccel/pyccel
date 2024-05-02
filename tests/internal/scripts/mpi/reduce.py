# coding: utf-8

from pyccel.stdlib.internal.mpi import mpi_init
from pyccel.stdlib.internal.mpi import mpi_finalize
from pyccel.stdlib.internal.mpi import mpi_comm_size
from pyccel.stdlib.internal.mpi import mpi_comm_rank
from pyccel.stdlib.internal.mpi import mpi_comm_world
from pyccel.stdlib.internal.mpi import mpi_status_size
from pyccel.stdlib.internal.mpi import mpi_reduce
from pyccel.stdlib.internal.mpi import MPI_INTEGER
from pyccel.stdlib.internal.mpi import MPI_SUM

# we need to declare these variables somehow,
# since we are calling mpi subroutines
ierr = -1
size = -1
rank = -1

mpi_init(ierr)

comm = mpi_comm_world
mpi_comm_size(comm, size, ierr)
mpi_comm_rank(comm, rank, ierr)

root = 0

if rank == 0:
    value = 1000
else:
    value = rank

sum_value = 0

mpi_reduce (value, sum_value, 1, MPI_INTEGER, MPI_SUM, root, comm, ierr)

if rank == 0:
    print('I, process ', root,', have the global sum value ', sum_value)

mpi_finalize(ierr)
