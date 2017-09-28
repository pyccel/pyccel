# coding: utf-8

#$ header procedure mpi_init()     results(int)
#$ header procedure mpi_finalize() results(int)
mpi_init     = eval('mpi_init')
mpi_finalize = eval('mpi_finalize')

ierr = mpi_init()

comm = mpi_comm_world
print("mpi_comm = ", comm)

#$ header procedure mpi_comm_size(int) results(int, int)
mpi_comm_size = eval('mpi_comm_size')
size, ierr = mpi_comm_size(comm)
print("mpi_size = ", size)

#$ header procedure mpi_comm_rank(int) results(int, int)
mpi_comm_rank = eval('mpi_comm_rank')
rank, ierr = mpi_comm_rank(comm)
print("mpi_rank = ", rank)

ierr = mpi_finalize()
