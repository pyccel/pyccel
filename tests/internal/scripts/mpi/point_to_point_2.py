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

    nx = np.int32(4)
    ny = np.int32(3 * 2)
    x = np.zeros(nx)
    y = np.zeros((3,2))

    if rank == 0:
        x[:] = 1.0
        y[:,:] = 1.0

    source = np.int32(0)
    dest   = np.int32(1)
    status = np.zeros(mpi_status_size, 'int32')

    # ...
    tag1 = np.int32(1234)
    if rank == source:
        mpi_send(x, nx, MPI_REAL8, dest, tag1, comm, ierr)
        print("> test 1: processor ", rank, " sent ", x)

    if rank == dest:
        mpi_recv(x, nx, MPI_REAL8, source, tag1, comm, status, ierr)
        print("> test 1: processor ", rank, " got  ", x)
    # ...

    # ...
    tag2 = np.int32(5678)
    count = np.int32(1)
    if rank == source:
        x[:] = 0.0
        x[1] = 2.0
        mpi_send(x[1], count, MPI_REAL8, dest, tag2, comm, ierr)
        print("> test 2: processor ", rank, " sent ", x[1])

    if rank == dest:
        mpi_recv(x[1], count, MPI_REAL8, source, tag2, comm, status, ierr)
        print("> test 2: processor ", rank, " got  ", x[1])
    # ...

    # ...
    tag3 = np.int32(4321)
    if rank == source:
        mpi_send(y, ny, MPI_REAL8, dest, tag3, comm, ierr)
        print("> test 3: processor ", rank, " sent ", y)

    if rank == dest:
        mpi_recv(y, ny, MPI_REAL8, source, tag3, comm, status, ierr)
        print("> test 3: processor ", rank, " got  ", y)
    # ...

    # ...
    tag4 = np.int32(8765)
    count = np.int32(1)
    if rank == source:
        y[:,:] = 0.0
        y[1,1] = 2.0
        mpi_send(y[1,1], count, MPI_REAL8, dest, tag4, comm, ierr)
        print("> test 4: processor ", rank, " sent ", y[1,1])

    if rank == dest:
        mpi_recv(y[1,1], count, MPI_REAL8, source, tag4, comm, status, ierr)
        print("> test 4: processor ", rank, " got  ", y[1,1])
    # ...

    # ...
    tag5 = np.int32(6587)
    count = np.int32(2)
    if rank == source:
        mpi_send(y[1,:], count, MPI_REAL8, dest, tag5, comm, ierr)
        print("> test 5: processor ", rank, " sent ", y[1,:])

    if rank == dest:
        mpi_recv(y[1,:], count, MPI_REAL8, source, tag5, comm, status, ierr)
        print("> test 5: processor ", rank, " got  ", y[1,:])
    # ...

    mpi_finalize(ierr)
