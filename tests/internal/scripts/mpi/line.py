# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

from pyccel.stdlib.internal.mpi import mpi_init
from pyccel.stdlib.internal.mpi import mpi_finalize
from pyccel.stdlib.internal.mpi import mpi_comm_size
from pyccel.stdlib.internal.mpi import mpi_comm_rank
from pyccel.stdlib.internal.mpi import mpi_comm_world
from pyccel.stdlib.internal.mpi import mpi_status_size
from pyccel.stdlib.internal.mpi import mpi_recv
from pyccel.stdlib.internal.mpi import mpi_send
from pyccel.stdlib.internal.mpi import mpi_type_vector
from pyccel.stdlib.internal.mpi import mpi_type_commit
from pyccel.stdlib.internal.mpi import mpi_type_free
from pyccel.stdlib.internal.mpi import MPI_INTEGER8

import numpy as np

# we need to declare these variables somehow,
# since we are calling mpi subroutines
ierr = np.int32(-1)
size = np.int32(-1)
rank = np.int32(-1)

mpi_init(ierr)

comm = mpi_comm_world

mpi_comm_size (comm, size, ierr)
mpi_comm_rank (comm, rank, ierr)

nb_lines   = np.int32(3)
nb_columns = np.int32(4)
tag        = np.int32(100)

a      = np.zeros ((nb_lines, nb_columns), 'int')
status = np.zeros (mpi_status_size, 'int32')

# Initialization of the matrix on each process
a[:,:] = 1000 + rank

# Definition of the type_line datatype
blocklength = np.int32(1)
type_line = np.int32(-1)
mpi_type_vector (nb_columns, blocklength, nb_lines, MPI_INTEGER8, type_line, ierr)

# Validation of the type_line datatype
mpi_type_commit (type_line, ierr)

# Sending of the first column
if ( rank == 0 ):
    dest = np.int32(1)
    mpi_send (a[1,0], nb_columns, MPI_INTEGER8, dest, tag, comm , ierr)

# Reception in the last column
if ( rank == 1 ):
    count  = np.int32(1)
    source = np.int32(0)
    mpi_recv (a[nb_lines-1,0], count, type_line, source, tag, comm, status, ierr)

print('I process ', rank, ', has a = ', a)

# Free the datatype
mpi_type_free (type_line, ierr)

mpi_finalize(ierr)
