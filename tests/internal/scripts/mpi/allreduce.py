# pylint: disable=missing-function-docstring, missing-module-docstring
# coding: utf-8

import numpy as np

from pyccel.stdlib.internal.mpi import (
    MPI_INTEGER8,
    MPI_PROD,
    mpi_allreduce,
    mpi_comm_rank,
    mpi_comm_size,
    mpi_comm_world,
    mpi_finalize,
    mpi_init,
)

if __name__ == "__main__":
    # we need to declare these variables somehow,
    # since we are calling mpi subroutines
    ierr = np.int32(-1)
    sizes = np.int32(-1)
    rank = np.int32(-1)

    mpi_init(ierr)

    comm = mpi_comm_world
    mpi_comm_size(comm, sizes, ierr)
    mpi_comm_rank(comm, rank, ierr)

    if rank == 0:
        value = 1000
    else:
        value = int(rank)

    product_value = 0
    length = np.int32(1)
    mpi_allreduce(value, product_value, length, MPI_INTEGER8, MPI_PROD, comm, ierr)

    print("I, process ", rank, ", have the global product value ", product_value)

    mpi_finalize(ierr)
