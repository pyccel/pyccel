# pylint: disable=missing-function-docstring, missing-module-docstring
# coding: utf-8

from pyccel.stdlib.internal.mpi import mpi_init
from pyccel.stdlib.internal.mpi import mpi_finalize
from pyccel.stdlib.internal.mpi import mpi_comm_size
from pyccel.stdlib.internal.mpi import mpi_comm_rank
from pyccel.stdlib.internal.mpi import mpi_comm_world
from pyccel.stdlib.internal.mpi import mpi_status_size
from pyccel.stdlib.internal.mpi import mpi_allgather
from pyccel.stdlib.internal.mpi import MPI_INTEGER8

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

    nb_values = 8

    block_length = np.int32(nb_values // sizes)

    # ...
    values = np.zeros(block_length, 'int')
    for i in range(0, block_length):
        values[i] = 1000 + rank*nb_values + i

    print('I, process ', rank, 'sent my values array : ', values)
    # ...

    # ...
    data = np.zeros(nb_values, 'int')

    mpi_allgather (values, block_length, MPI_INTEGER8,
                   data, block_length, MPI_INTEGER8,
                   comm, ierr)
    # ...

    print('I, process ', rank, ', received ', data)

    mpi_finalize(ierr)
