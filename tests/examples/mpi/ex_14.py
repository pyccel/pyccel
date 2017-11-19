# coding: utf-8

#must be run with mpirun -n 4

from pyccel.mpi import *

ierr = mpi_init()

comm = mpi_comm_world
size = comm.size
rank = comm.rank

north = 0
east  = 1
south = 2
west  = 3

neighbor = zeros(4, int)
coords   = zeros(2, int)

dims    = array((2, 2), int)
periods = array((False, True), bool)

reorder = False
ierr = comm.cart_create(dims, periods, reorder, comm_2d)

#Know my coordinates in the topology
rank_in_topo = comm_2d.rank
ierr = comm_2d.cart_coords(rank_in_topo, coords)

#Search of my West and East neigbors
ierr = comm_2d.cart_shift(0, 1, neighbor(west), neighbor(east))

#Search of my South and North neighbors
ierr = comm_2d.cart_shift(1, 1, neighbor(south), neighbor(north))

print(("I, ", rank_in_topo, " process, has neighbor :", neighbor))

#Destruction of the communicators
ierr = comm_2d.free()

ierr = mpi_finalize()
