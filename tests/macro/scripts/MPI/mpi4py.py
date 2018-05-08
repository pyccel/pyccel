#$ header class MPI_(public,hide)
#$ header method __init__(MPI_)



from pyccel.stdlib.internal.mpi import mpi_comm_world
from pyccel.stdlib.internal.mpi import mpi_comm_rank
from pyccel.stdlib.internal.mpi import mpi_comm_size
from pyccel.stdlib.internal.mpi import mpi_send
from pyccel.stdlib.internal.mpi import mpi_recv
from pyccel.stdlib.internal.mpi import mpi_comm_split
from pyccel.stdlib.internal.mpi import mpi_bcast
from pyccel.stdlib.internal.mpi 	import mpi_comm_free
from numpy import zeros

class MPI_:
    def __init__(self):
        self.COMM_WORLD = 0





MPI = MPI_()

#$ header macro x.COMM_WORLD := mpi_comm_world
#$ header macro (x), y.Get_rank() := mpi_comm_rank(y,x,ierr)
#$ header macro (x), y.Get_size() := mpi_comm_size(y,x,ierr)
#$ header macro  y.send(data, dest, tag=0) := mpi_send(data, data.count, data.dtype, dest ,tag, y, ierr)
#$ header macro (x), y.recv(source=0, tag=0) := mpi_recv(x, x.count, x.dtype, source ,tag, y, status, ierr)
#$ header macro (x), y.Split(color=0,key=0) := mpi_comm_split(y, color, key, x, ierr)
#$ header macro  y.bcast(data, root=0) := mpi_bcast(data, data.count, data.dtype, root, y, ierr)
#$ header macro y.Free() := mpi_comm_free(y, ierr)


