# coding: utf-8

from mpi import mpi_init
from mpi import mpi_finalize
from mpi import mpi_comm_size
from mpi import mpi_comm_rank
from mpi import mpi_comm_world

# we need to declare these variables somehow,
# since we are calling mpi subroutines
ierr = -1
size = -1
rank = -1

mpi_init(ierr)

comm = mpi_comm_world

mpi_comm_size(comm, size, ierr)
mpi_comm_rank(comm, rank, ierr)

print('I process ', rank, ', among ', size, ' processes')

mpi_finalize(ierr)
