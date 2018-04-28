# coding: utf-8

from pyccel.stdlib.internal.mpi import mpi_init
from pyccel.stdlib.internal.mpi import mpi_finalize
from pyccel.stdlib.internal.mpi import mpi_comm_size
from pyccel.stdlib.internal.mpi import mpi_comm_rank
from pyccel.stdlib.internal.mpi import mpi_comm_world
from pyccel.stdlib.internal.mpi import mpi_status_size
from pyccel.stdlib.internal.mpi import mpi_comm_split
from pyccel.stdlib.internal.mpi import mpi_comm_free
from pyccel.stdlib.internal.mpi import mpi_bcast
from pyccel.stdlib.internal.mpi import MPI_INTEGER

from numpy import zeros

# we need to declare these variables somehow,
# since we are calling mpi subroutines
ierr = -1
size = -1
rank_in_world = -1

mpi_init(ierr)

comm = mpi_comm_world
mpi_comm_size(comm, size, ierr)
mpi_comm_rank(comm, rank_in_world, ierr)

master = 0
m      = 8

a = zeros(m, 'int')

if rank_in_world == 1:
    a[:] = 1
if rank_in_world == 2:
    a[:] = 2

key = rank_in_world
if rank_in_world == 1:
    key = -1
if rank_in_world == 2:
    key = -1

two   = 2
color = rank_in_world % two

newcomm = -1
mpi_comm_split (comm, color, key, newcomm, ierr)

# Broadcast of the message by the rank process master of
# each communicator to the processes of its group
mpi_bcast (a, m, MPI_INTEGER, master, newcomm, ierr)

print("> processor ", rank_in_world, " has a = ", a)

# Destruction of the communicators
mpi_comm_free (newcomm, ierr)

mpi_finalize(ierr)
