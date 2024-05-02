# pylint: disable=missing-function-docstring, missing-module-docstring
# coding: utf-8

from pyccel.stdlib.internal.mpi import mpi_init
from pyccel.stdlib.internal.mpi import mpi_finalize
from pyccel.stdlib.internal.mpi import mpi_comm_size
from pyccel.stdlib.internal.mpi import mpi_comm_rank
from pyccel.stdlib.internal.mpi import mpi_comm_world
from pyccel.stdlib.internal.mpi import mpi_status_size
from pyccel.stdlib.internal.mpi import mpi_send
from pyccel.stdlib.internal.mpi import mpi_recv
from pyccel.stdlib.internal.mpi import MPI_REAL8

import numpy as np

if __name__ == '__main__':
    # we need to declare these variables somehow,
    # since we are calling mpi subroutines
    ierr = np.int32(-1)
    sizes = np.int32(-1)
    rank = np.int32(-1)

    mpi_init(ierr)

    comm = mpi_comm_world
    mpi_comm_size(comm, sizes, ierr)
    mpi_comm_rank(comm, rank, ierr)

    nx = 4
    x = np.zeros(nx)

    if rank == 0:
        x[:] = 1.0

    source = np.int32(0)
    dest   = np.int32(1)

    # ...
    tag1 = np.int32(1234)
    if rank == source:
        x[1] = 2.0
        count = np.int32(1)
        mpi_send(x[1], count, MPI_REAL8, dest, tag1, comm, ierr)
        print("> processor ", rank, " sent x(1) = ", x)
    # ...

    mpi_finalize(ierr)
