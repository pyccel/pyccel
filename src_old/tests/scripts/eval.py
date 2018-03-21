# coding: utf-8

#$ header procedure mpi_init()     results(int)
#$ header procedure mpi_finalize() results(int)
mpi_init     = eval('mpi_init')
mpi_finalize = eval('mpi_finalize')

ierr = mpi_init()

comm = mpi_comm_world
print(("mpi_comm = ", comm))

#$ header procedure mpi_comm_size(int) results(int, int)
mpi_comm_size = eval('mpi_comm_size')
size, ierr = mpi_comm_size(comm)
print(("mpi_size = ", size))

#$ header procedure mpi_comm_rank(int) results(int, int)
mpi_comm_rank = eval('mpi_comm_rank')
rank, ierr = mpi_comm_rank(comm)
print(("mpi_rank = ", rank))

tag         = 1234
source      = 0
destination = 1
count       = 1
status      = -1
#$ header procedure mpi_send(int, int, int, int, int, int)      results(int)
#$ header procedure mpi_recv(int, int, int, int, int, int, int) results(int)
mpi_send = eval('mpi_send')
mpi_recv = eval('mpi_recv')
#if rank == source:
buffer = 5678
ierr = mpi_send(buffer, count, mpi_integer, destination, tag, mpi_comm_world)
print(("processor ", rank, " sent ", buffer))
#if rank == destination:
#ierr = mpi_recv(buffer, count, mpi_integer, source, tag, mpi_comm_world, status)
#print("processor ", rank, " got ", buffer)

ierr = mpi_finalize()
