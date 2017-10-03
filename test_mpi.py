# coding: utf-8

ierr = mpi_init()

comm = mpi_comm_world
size = comm.size
rank = comm.rank

x = zeros(4, double)

source = 0
dest   = 1
if rank == 0:
    x = 1.0

if rank == source:
    ierr = comm.send(x, rank, dest)
    print("processor ", rank, " sent ", x)
if rank == dest:
    print("recv")

ierr = mpi_finalize()
