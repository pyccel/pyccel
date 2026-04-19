# pylint: disable=missing-function-docstring, missing-module-docstring
# coding: utf-8

import numpy as np

from pyccel.stdlib.internal.mpi import (
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

    print("I process ", rank, ", among ", sizes, " processes")

    mpi_finalize(ierr)
