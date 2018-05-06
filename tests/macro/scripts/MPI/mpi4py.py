#$ header class MPI_(public,hide)
#$ header method __init__(MPI_)



from pyccel.stdlib.internal.mpi import mpi_comm_world
from pyccel.stdlib.internal.mpi import mpi_comm_rank

class MPI_:
    def __init__(self):
        self.COMM_WORLD = 0





MPI = MPI_()

#$ header macro x.COMM_WORLD := mpi_comm_world
#$ header macro (x), y.Get_rank() := mpi_comm_rank(y,x,ierr|ierr)

