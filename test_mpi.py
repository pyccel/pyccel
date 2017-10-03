# coding: utf-8

ierr = mpi_init()

comm = mpi_comm_world
#print("mpi_comm = ", comm)

size = comm.size
#print("mpi_size = ", size)

rank = comm.rank
#print("mpi_rank = ", rank)

x = zeros(4, double)

if rank == 0:
    x = 1.0

ierr = comm.send(x)

print('PROC ', rank, ' x = ', x)

if rank == 0:
    print('That is all for now!')

ierr = mpi_finalize()
