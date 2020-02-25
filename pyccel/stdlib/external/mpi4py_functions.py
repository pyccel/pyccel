from numpy import int32

from pyccel.stdlib.internal.mpi import mpi_comm_rank
from pyccel.stdlib.internal.mpi import mpi_comm_size

#===================================================================================

@types('int')
def mpi4py_get_rank(comm):
    rank = int32(-1)
    ierr = int32(-1)
    mpi_comm_rank(int32(comm), rank, ierr)
    return int(rank)

@types('int')
def mpi4py_get_size(comm):
    size = int32(-1)
    ierr = int32(-1)
    mpi_comm_size(int32(comm), size, ierr)
    return int(size)
