# pylint: disable=missing-function-docstring, missing-module-docstring
# coding: utf-8

from pyccel.stdlib.internal.mpi import mpi_init
from pyccel.stdlib.internal.mpi import mpi_finalize
from pyccel.stdlib.internal.mpi import mpi_comm_size
from pyccel.stdlib.internal.mpi import mpi_comm_rank
from pyccel.stdlib.internal.mpi import mpi_comm_world
from pyccel.stdlib.internal.mpi import mpi_status_size
from pyccel.stdlib.internal.mpi import mpi_scatter
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

    master    = np.int32(1)
    nb_values = np.int32(8)

    block_length = nb_values // sizes

    data = np.zeros(block_length, 'int')

    if rank == master:
        values = np.zeros(nb_values, 'int')
        for i in range(0, nb_values):
            values[i] = 1000 + i

        print('I, process ', rank ,' send my values array', values)

    mpi_scatter (values, block_length, MPI_INTEGER8,
                 data,   block_length, MPI_INTEGER8,
                 master, comm, ierr)

    print('I, process ', rank, ', received ', data, ' of process ', master)

    mpi_finalize(ierr)
