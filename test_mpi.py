# coding: utf-8

ierr = mpi_init()

comm = mpi_comm_world
print("mpi_comm = ", comm)

size = comm.size
print("mpi_size = ", size)

rank = comm.rank
print("mpi_rank = ", rank)

ierr = mpi_finalize()
