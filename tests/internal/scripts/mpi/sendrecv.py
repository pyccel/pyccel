# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

from pyccel.stdlib.internal.mpi import mpi_init
from pyccel.stdlib.internal.mpi import mpi_finalize
from pyccel.stdlib.internal.mpi import mpi_comm_size
from pyccel.stdlib.internal.mpi import mpi_comm_rank
from pyccel.stdlib.internal.mpi import mpi_comm_world
from pyccel.stdlib.internal.mpi import mpi_status_size
from pyccel.stdlib.internal.mpi import mpi_sendrecv
from pyccel.stdlib.internal.mpi import MPI_INTEGER8

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

if rank == 0:
    partner = np.int32(1)

if rank == 1:
    partner = np.int32(0)

msg = rank + 1000
val = -1

sendcount = np.int32(1)
recvcount = np.int32(1)
tag = np.int32(1234)
status = np.zeros(mpi_status_size, 'int32')

mpi_sendrecv (msg, sendcount, MPI_INTEGER8, partner, tag,
              val, recvcount, MPI_INTEGER8, partner, tag,
              comm, status, ierr)

print('I, process ', rank, ', I received', val, ' from process ', partner)

mpi_finalize(ierr)
