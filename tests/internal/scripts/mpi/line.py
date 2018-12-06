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
from pyccel.stdlib.internal.mpi import MPI_INTEGER

from numpy import zeros

# we need to declare these variables somehow,
# since we are calling mpi subroutines
ierr = -1
size = -1
rank = -1

mpi_init(ierr)

comm = mpi_comm_world

mpi_comm_size (comm, size, ierr)
mpi_comm_rank (comm, rank, ierr)

nb_lines   = 3
nb_columns = 4
tag        = 100

a      = zeros ((nb_lines, nb_columns), 'int')
status = zeros (mpi_status_size, 'int')

# Initialization of the matrix on each process
a[:,:] = 1000 + rank

# Definition of the type_line datatype
type_line = -1
mpi_type_vector (nb_columns, 1, nb_lines, MPI_INTEGER, type_line, ierr)

# Validation of the type_line datatype
mpi_type_commit (type_line, ierr)

# Sending of the first column
if ( rank == 0 ):
    mpi_send (a[1,0], nb_columns, MPI_INTEGER, 1, tag, comm , ierr)

# Reception in the last column
if ( rank == 1 ):
    mpi_recv (a[nb_lines-1,0], 1, type_line, 0, tag, comm, status, ierr)

print('I process ', rank, ', has a = ', a)

# Free the datatype
mpi_type_free (type_line, ierr)

mpi_finalize(ierr)
