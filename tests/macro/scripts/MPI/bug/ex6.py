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
from pyccel.stdlib.internal.mpi import MPI_INTEGER

from numpy import zeros

# we need to declare these variables somehow,
# since we are calling mpi subroutines

size = -1
rank = -1



comm = mpi_comm_world

mpi_comm_size (comm, size, ierr)
mpi_comm_rank (comm, rank, ierr)

nb_lines   = 3
nb_columns = 4
tag        = 100

a      = zeros ((nb_lines, nb_columns), 'int')


# Initialization of the matrix on each process
a = 1000 + rank

# Definition of the type_line datatype
type_line = -1
type_line = MPI.INT.Create_vector(nb_columns, 1, nb_lines)

# Validation of the type_line datatype
type_line.Commit()

# Sending of the first column
if ( rank == 0 ):
    comm.send(a[1,0],1, tag)

# Reception in the last column
if ( rank == 1 ):
    a[nb_lines-1,0]=comm.recv(a[nb_lines-1,0], 1, type_line, 0, tag, comm, status, ierr)

print('I process ', rank, ', has a = ', a)

# Free the datatype
mpi_type_free (type_line, ierr)

