# coding: utf-8

from pyccel.stdlib.internal.mpi import mpi_init
from pyccel.stdlib.internal.mpi import mpi_finalize
from pyccel.stdlib.internal.mpi import mpi_comm_size
from pyccel.stdlib.internal.mpi import mpi_comm_rank
from pyccel.stdlib.internal.mpi import mpi_comm_world
from pyccel.stdlib.internal.mpi import mpi_status_size
from pyccel.stdlib.internal.mpi import mpi_send
from pyccel.stdlib.internal.mpi import mpi_recv
from pyccel.stdlib.internal.mpi import MPI_DOUBLE

from numpy import zeros

# we need to declare these variables somehow,
# since we are calling mpi subroutines
ierr = -1
size = -1
rank = -1

mpi_init(ierr)

comm = mpi_comm_world
mpi_comm_size(comm, size, ierr)
mpi_comm_rank(comm, rank, ierr)

nx = 4
ny = 3 * 2
x = zeros(nx)
y = zeros((3,2))

if rank == 0:
    x[:] = 1.0
    y[:,:] = 1.0

source = 0
dest   = 1
status = zeros(mpi_status_size, 'int')

# ...
tag1 = 1234
if rank == source:
    mpi_send(x, nx, MPI_DOUBLE, dest, tag1, comm, ierr)
    print("> test 1: processor ", rank, " sent ", x)

if rank == dest:
    mpi_recv(x, nx, MPI_DOUBLE, source, tag1, comm, status, ierr)
    print("> test 1: processor ", rank, " got  ", x)
# ...

# ...
tag2 = 5678
if rank == source:
    x[:] = 0.0
    x[1] = 2.0
    mpi_send(x[1], 1, MPI_DOUBLE, dest, tag2, comm, ierr)
    print("> test 2: processor ", rank, " sent ", x[1])

if rank == dest:
    mpi_recv(x[1], 1, MPI_DOUBLE, source, tag2, comm, status, ierr)
    print("> test 2: processor ", rank, " got  ", x[1])
# ...

# ...
tag3 = 4321
if rank == source:
    mpi_send(y, ny, MPI_DOUBLE, dest, tag3, comm, ierr)
    print("> test 3: processor ", rank, " sent ", y)

if rank == dest:
    mpi_recv(y, ny, MPI_DOUBLE, source, tag3, comm, status, ierr)
    print("> test 3: processor ", rank, " got  ", y)
# ...

# ...
tag4 = 8765
if rank == source:
    y[:,:] = 0.0
    y[1,1] = 2.0
    mpi_send(y[1,1], 1, MPI_DOUBLE, dest, tag4, comm, ierr)
    print("> test 4: processor ", rank, " sent ", y[1,1])

if rank == dest:
    mpi_recv(y[1,1], 1, MPI_DOUBLE, source, tag4, comm, status, ierr)
    print("> test 4: processor ", rank, " got  ", y[1,1])
# ...

# ...
tag5 = 6587
if rank == source:
    mpi_send(y[1,:], 2, MPI_DOUBLE, dest, tag5, comm, ierr)
    print("> test 5: processor ", rank, " sent ", y[1,:])

if rank == dest:
    mpi_recv(y[1,:], 2, MPI_DOUBLE, source, tag5, comm, status, ierr)
    print("> test 5: processor ", rank, " got  ", y[1,:])
# ...

mpi_finalize(ierr)
