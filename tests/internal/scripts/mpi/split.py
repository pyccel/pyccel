# pylint: disable=missing-function-docstring, missing-module-docstring
# coding: utf-8

import numpy as np

from pyccel.stdlib.internal.mpi import (
    MPI_INTEGER8,
    mpi_bcast,
    mpi_comm_free,
    mpi_comm_rank,
    mpi_comm_size,
    mpi_comm_split,
    mpi_comm_world,
    mpi_finalize,
    mpi_init,
)

if __name__ == "__main__":
    # we need to declare these variables somehow,
    # since we are calling mpi subroutines
    ierr = np.int32(-1)
    sizes = np.int32(-1)
    rank_in_world = np.int32(-1)

    mpi_init(ierr)

    comm = mpi_comm_world
    mpi_comm_size(comm, sizes, ierr)
    mpi_comm_rank(comm, rank_in_world, ierr)

    master = np.int32(0)
    m = np.int32(8)

    a = np.zeros(m, "int")

    if rank_in_world == 1:
        a[:] = 1
    if rank_in_world == 2:
        a[:] = 2

    key = rank_in_world
    if rank_in_world == 1:
        key = np.int32(-1)
    if rank_in_world == 2:
        key = np.int32(-1)

    two = 2
    c = rank_in_world % two

    color = np.int32(c)
    newcomm = np.int32(-1)
    mpi_comm_split(comm, color, key, newcomm, ierr)

    # Broadcast of the message by the rank process master of
    # each communicator to the processes of its group
    mpi_bcast(a, m, MPI_INTEGER8, master, newcomm, ierr)

    print("> processor ", rank_in_world, " has a = ", a)

    # Destruction of the communicators
    mpi_comm_free(newcomm, ierr)

    mpi_finalize(ierr)
