from mpi4py import MPI


#$ header variable ierr  int 

# we must add this header to declare ierr 
# this variable name must not be used in the code
rank = -1


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

data = [0,0]
if rank == 0:
    data = [7,4]
    comm.send(data, 1, tag=11)
elif rank == 1:
    data = comm.recv(source=0, tag=11)
