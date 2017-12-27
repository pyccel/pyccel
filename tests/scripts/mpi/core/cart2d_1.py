# coding: utf-8

from pyccel.stdlib.parallel.mpi import mpi_init
from pyccel.stdlib.parallel.mpi import mpi_finalize
from pyccel.stdlib.parallel.mpi import mpi_comm_size
from pyccel.stdlib.parallel.mpi import mpi_comm_rank
from pyccel.stdlib.parallel.mpi import mpi_comm_world
from pyccel.stdlib.parallel.mpi import mpi_status_size
from pyccel.stdlib.parallel.mpi import MPI_INTEGER
from pyccel.stdlib.parallel.mpi import mpi_cart_create
from pyccel.stdlib.parallel.mpi import mpi_cart_coords
from pyccel.stdlib.parallel.mpi import mpi_cart_shift
from pyccel.stdlib.parallel.mpi import mpi_comm_free

# we need to declare these variables somehow,
# since we are calling mpi subroutines
ierr = -1
size = -1

# rank in comm worl
rank = -1

# rank is 2d cart
rank_in_topo = -1

# 2d cart communicator
comm_2d = -1

mpi_init(ierr)

comm = mpi_comm_world
mpi_comm_size(comm, size, ierr)
mpi_comm_rank(comm, rank, ierr)

north = 0
east  = 1
south = 2
west  = 3

ndims   = 2
steps   = [1, 1]
dims    = [2, 2]
periods = [False, True]

reorder = False

neighbor = zeros(4, int)
coords   = zeros(2, int)

# Create a 2d mpi cart
mpi_cart_create (comm, ndims, dims, periods, reorder, comm_2d, ierr)

# Know my coordinates in the topology
mpi_comm_rank (comm_2d, rank_in_topo, ierr)
mpi_cart_coords (comm_2d, rank_in_topo, ndims, coords, ierr)

# Search of my West and East neigbors
mpi_cart_shift (comm_2d, 0, steps[0], neighbor[west], neighbor[east], ierr)

#Search of my South and North neighbors
mpi_cart_shift (comm_2d, 1, steps[1], neighbor[south], neighbor[north], ierr)

print("I, ", rank_in_topo, " process, has neighbor :", neighbor)

# Destruction of the communicators
mpi_comm_free (comm_2d, ierr)

mpi_finalize(ierr)
