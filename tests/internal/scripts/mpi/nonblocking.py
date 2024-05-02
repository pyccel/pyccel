# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

from pyccel.stdlib.internal.mpi import mpi_init
from pyccel.stdlib.internal.mpi import mpi_finalize
from pyccel.stdlib.internal.mpi import mpi_comm_size
from pyccel.stdlib.internal.mpi import mpi_comm_rank
from pyccel.stdlib.internal.mpi import mpi_comm_world
from pyccel.stdlib.internal.mpi import mpi_status_size
from pyccel.stdlib.internal.mpi import mpi_isend
from pyccel.stdlib.internal.mpi import mpi_irecv
from pyccel.stdlib.internal.mpi import mpi_waitall
from pyccel.stdlib.internal.mpi import MPI_REAL8

import numpy as np

# we need to declare these variables somehow,
# since we are calling mpi subroutines
ierr = np.int32(-1)
size = np.int32(-1)
rank = np.int32(-1)

mpi_init(ierr)

comm = mpi_comm_world
mpi_comm_size(comm, size, ierr)
mpi_comm_rank(comm, rank, ierr)

n = np.int32(4)
x = np.zeros(n)
y = np.zeros(n)

if rank == 0:
    x[:] = 1.0
    y[:] = 2.0

# ...
tag0 = np.int32(1234)
tag1 = np.int32(5678)
reqs = np.zeros(4, 'int32')
# ...

# ...
before = np.int32(rank - 1)
after  = np.int32(rank + 1)
if rank == 0:
    before = np.int32(size - 1)
if rank == size - 1:
    after  = np.int32(0)

# ...
mpi_irecv(x, n, MPI_REAL8, before, tag0, comm, reqs[0], ierr)
mpi_irecv(y, n, MPI_REAL8, after , tag1, comm, reqs[1], ierr)

mpi_isend(x, n, MPI_REAL8, before, tag1, comm, reqs[2], ierr)
mpi_isend(y, n, MPI_REAL8, after , tag0, comm, reqs[3], ierr)
# ...

# ...
statuses = np.zeros((mpi_status_size, n), 'int32')
mpi_waitall(n, reqs, statuses, ierr)
# ...

mpi_finalize(ierr)
