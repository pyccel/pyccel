from mpi4py import MPI

#$ header variable ierr  int 

# we must add this header to declare ierr 
# this variable name must not be used in the code
rank = -1


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
