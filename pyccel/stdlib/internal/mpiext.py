from numpy import int32

from pyccel.stdlib.internal.mpi import mpi_comm_rank
from pyccel.stdlib.internal.mpi import mpi_comm_size

#===================================================================================
# TODO: check error code in 'ierr' and raise an error in Fortran if (ierr != 0)
# TODO: create class for communicator and change functions into members (as mpi4py)
# TODO: automatically initialize MPI when loading extension module

@types('int32')
def mpiext_get_rank(comm):
    rank = int32(-1)
    ierr = int32(-1)
    mpi_comm_rank(comm, rank, ierr)
    return int(rank)

@types('int32')
def mpiext_get_size(comm):
    size = int32(-1)
    ierr = int32(-1)
    mpi_comm_size(comm, size, ierr)
    return int(size)
