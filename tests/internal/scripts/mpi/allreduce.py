# coding: utf-8

from pyccel.stdlib.internal.mpi import mpi_init
from pyccel.stdlib.internal.mpi import mpi_finalize
from pyccel.stdlib.internal.mpi import mpi_comm_size
from pyccel.stdlib.internal.mpi import mpi_comm_rank
from pyccel.stdlib.internal.mpi import mpi_comm_world
from pyccel.stdlib.internal.mpi import mpi_status_size
from pyccel.stdlib.internal.mpi import mpi_allreduce
from pyccel.stdlib.internal.mpi import MPI_INTEGER
from pyccel.stdlib.internal.mpi import MPI_PROD

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
    value = 1000
else:
    value = rank

product_value = 0
mpi_allreduce (value, product_value, 1, MPI_INTEGER, MPI_PROD, comm, ierr)

print('I, process ', rank,', have the global product value ', product_value)

mpi_finalize(ierr)
