# coding: utf-8

from pyccel.mpi import *

ierr = mpi_init()

comm          = mpi_comm_world
size          = comm.size
rank_in_world = comm.rank

root = 0
m    = 8

a = zeros(m, double)

if rank_in_world == 1:
    a = 1.0
if rank_in_world == 2:
    a = 2.0

key = rank_in_world
if rank_in_world == 1:
    key = -1
if rank_in_world == 2:
    key = -1

two   = 2
color = mod(rank_in_world, two)

ierr = comm.split (color, key, newcomm)

#Broadcast of the message by the rank process 0 of
#each communicator to the processes of its group
ierr = newcomm.bcast (a, root)

#Destruction of the communicators
ierr = newcomm.free()

ierr = mpi_finalize()
