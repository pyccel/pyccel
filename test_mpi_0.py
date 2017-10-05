# coding: utf-8

ierr = mpi_init()

comm = mpi_comm_world
size = comm.size
rank = comm.rank

n = 4
x = zeros(n, double)

if rank == 0:
    x = 1.0

source = 0
dest   = 1
tagx = 1234
if rank == source:
    ierr = comm.send(x, dest, tagx)
    print("processor ", rank, " sent ", x)

if rank == dest:
    ierr = comm.recv(x, source, tagx)
    print("processor ", rank, " got  ", x)

tag1 = 5678
if rank == source:
    x[1] = 2.0
    ierr = comm.send(x[1], dest, tag1)
    print("processor ", rank, " sent x(1) = ", x[1])

if rank == dest:
    ierr = comm.recv(x[1], source, tag1)
    print("processor ", rank, " got  x(1) = ", x[1])

ierr = mpi_finalize()
