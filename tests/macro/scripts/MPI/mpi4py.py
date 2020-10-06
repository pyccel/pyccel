# pylint: disable=missing-function-docstring, missing-module-docstring/
#$ header class MPI_(public)
#$ header method __init__(MPI_)



from pyccel.stdlib.internal.mpi import mpi_comm_world
from pyccel.stdlib.internal.mpi import mpi_comm_rank
from pyccel.stdlib.internal.mpi import mpi_comm_size

from pyccel.stdlib.internal.mpi import mpi_send
from pyccel.stdlib.internal.mpi import mpi_ssend
from pyccel.stdlib.internal.mpi import mpi_bsend

from pyccel.stdlib.internal.mpi import mpi_isend
from pyccel.stdlib.internal.mpi import mpi_issend
from pyccel.stdlib.internal.mpi import mpi_ibsend

from pyccel.stdlib.internal.mpi import mpi_recv
from pyccel.stdlib.internal.mpi import mpi_irecv

from pyccel.stdlib.internal.mpi import mpi_sendrecv
from pyccel.stdlib.internal.mpi import mpi_sendrecv_replace

from pyccel.stdlib.internal.mpi import mpi_bcast
from pyccel.stdlib.internal.mpi import mpi_scatter

from pyccel.stdlib.internal.mpi import mpi_barrier
from pyccel.stdlib.internal.mpi import mpi_gather
from pyccel.stdlib.internal.mpi import mpi_allgather
from pyccel.stdlib.internal.mpi import mpi_gatherv
from pyccel.stdlib.internal.mpi import mpi_alltoall

from pyccel.stdlib.internal.mpi import mpi_reduce
from pyccel.stdlib.internal.mpi import mpi_allreduce

from pyccel.stdlib.internal.mpi import mpi_wait
from pyccel.stdlib.internal.mpi import mpi_waitall
from pyccel.stdlib.internal.mpi import mpi_waitany
from pyccel.stdlib.internal.mpi import mpi_waitsome

from pyccel.stdlib.internal.mpi import mpi_test
from pyccel.stdlib.internal.mpi import mpi_testall
from pyccel.stdlib.internal.mpi import mpi_testany
from pyccel.stdlib.internal.mpi import mpi_testsome

from pyccel.stdlib.internal.mpi import mpi_cart_create
from pyccel.stdlib.internal.mpi import mpi_cart_coords
from pyccel.stdlib.internal.mpi import mpi_cart_shift
from pyccel.stdlib.internal.mpi import mpi_cart_sub

from pyccel.stdlib.internal.mpi import mpi_comm_split
from pyccel.stdlib.internal.mpi import mpi_comm_free

from pyccel.stdlib.internal.mpi import mpi_type_vector
from pyccel.stdlib.internal.mpi import mpi_type_commit
from pyccel.stdlib.internal.mpi import mpi_type_contiguous
from pyccel.stdlib.internal.mpi import mpi_type_free
from pyccel.stdlib.internal.mpi import mpi_type_indexed
from pyccel.stdlib.internal.mpi import mpi_type_create_subarray
from pyccel.stdlib.internal.mpi import ANY_TAG
from pyccel.stdlib.internal.mpi import ANY_SOURCE

from pyccel.stdlib.internal.mpi import MPI_SUM
from pyccel.stdlib.internal.mpi import MPI_PROD
from pyccel.stdlib.internal.mpi import MPI_MAX
from pyccel.stdlib.internal.mpi import MPI_MIN
from pyccel.stdlib.internal.mpi import MPI_MAXLOC
from pyccel.stdlib.internal.mpi import MPI_MINLOC
from pyccel.stdlib.internal.mpi import MPI_LAND
from pyccel.stdlib.internal.mpi import MPI_LOR
from pyccel.stdlib.internal.mpi import MPI_LXOR
from pyccel.stdlib.internal.mpi import MPI_INTEGER
from pyccel.stdlib.internal.mpi import MPI_DOUBLE


class MPI_:
    def __init__(self):
        self.COMM_WORLD = 0
        self.INT        = MPI_INTEGER
        self.DOUBLE     = MPI_DOUBLE
        self.SUM        = MPI_SUM
        self.PROD       = MPI_PROD
        self.MAX        = MPI_MAX
        self.MIN        = MPI_MIN
        self.MAXLOC     = MPI_MAXLOC
        self.MINLOC     = MPI_MINLOC
        self.LAND       = MPI_LAND
        self.LOR        = MPI_LOR
        self.LXOR       = MPI_LXOR



MPI = MPI_()

#$ header macro x.COMM_WORLD := mpi_comm_world
#$ header macro x.SUM        := MPI_SUM
#$ header macro x.PROD       := MPI_PROD
#$ header macro x.MAX        := MPI_MAX
#$ header macro x.MIN        := MPI_MIN
#$ header macro x.MAXLOC     := MPI_MAXLOC
#$ header macro x.MINLOC     := MPI_MINLOC
#$ header macro x.LAND       := MPI_LAND
#$ header macro x.LOR        := MPI_LOR
#$ header macro x.LXOR       := MPI_LXOR
#$ header macro x.INT        := MPI_INTEGER
#$ header macro x.DOUBLE     := MPI_DOUBLE


#$ header macro (x), y.Get_rank() := mpi_comm_rank(y,x,ierr)
#$ header macro (x), y.Get_size() := mpi_comm_size(y,x,ierr)


#......................
#lower-case letter functions
#......................

#$ header macro  y.send(data, dest, tag=0)  := mpi_send(data, data.count, data.dtype, dest ,tag, y, ierr)
#$ header macro  y.ssend(data, dest, tag=0) := mpi_ssend(data, data.count, data.dtype, dest ,tag, y, ierr)
#$ header macro  y.bsend(data, dest, tag=0) := mpi_bsend(data, data.count, data.dtype, dest ,tag, y, ierr)

#$ header macro  (req),y.isend(data, dest, tag=0)  := mpi_isend(data, data.count, data.dtype, dest ,tag, y, req, ierr)
#$ header macro  (req),y.issend(data, dest, tag=0) := mpi_issend(data, data.count, data.dtype, dest ,tag, y, req, ierr)
#$ header macro  (req),y.ibsend(data, dest, tag=0) := mpi_ibsend(data, data.count, data.dtype, dest ,tag, y, req, ierr)

#$ header macro (x), y.recv(source=0, tag=0) := mpi_recv(x, x.count, x.dtype, source ,tag, y, status, ierr)

#$ header macro (x), y.sendrecv(sendobj, dest, sendtag=0, source=ANY_SOURCE, recvtag=ANY_TAG) := mpi_sendrecv(sendobj, sendobj.count, sendobj.dtype,  dest, sendtag, x, x.count, x.dtype, source , recvtag, y, status, ierr)

#$ header macro (x),y.reduce(data, op=MPI_SUM, root=0) := mpi_reduce(data, x, data.count, data.dtype, op ,root, y, ierr)
#$ header macro (x),y.allreduce(data, op=MPI_SUM) := mpi_allreduce(data, x, data.count, data.dtype, op , y, ierr)

#$ header macro  y.bcast(data, root=0) := mpi_bcast(data, data.count, data.dtype, root, y, ierr)
#$ header macro  (x),y.gather(data, root=0) := mpi_gather(data, data.count, data.dtype, x, x.count, x.dtype, root, y, ierr)

#.....................
##$ header macro (x),y.scatter
##$ header macro (req), y.irecv
##$ header macro y.alltoall
#not_working for the moment
#.....................


#......................
#upper-case letter functions
#......................

#$ header macro (x), y.Split(color=0, key=0) := mpi_comm_split(y, color, key, x, ierr)
#$ header macro y.Free() := mpi_comm_free(y, ierr)
#$ header macro (datatype),y.Create_vector(count, blocklength, stride) := mpi_type_vector(count, blocklength, stride, y.dtype, datatype, ierr)
#$ header macro x.Commit() := mpi_type_commit(x,ierr)


#$ header macro  y.Send([data, dtype=data.dtype], dest=0, tag=0)  := mpi_send(data, data.count, dtype, dest ,tag, y, ierr)
#$ header macro  y.Recv([data, dtype=data.dtype], source=ANY_SOURCE, tag=ANY_TAG) := mpi_recv(data, data.count, data.dtype, source ,tag, y, status, ierr)

#$ header macro  (req),y.Isend([data, dtype=data.dtype], dest=0, tag=0)  := mpi_isend(data, data.count, dtype, dest ,tag, y, req, ierr)
#$ header macro  (req),y.Issend([data, dtype=data.dtype], dest=0, tag=0)  := mpi_issend(data, data.count, dtype, dest ,tag, y, ierr)
#$ header macro  (req),y.Ibsend([data, dtype=data.dtype], dest=0, tag=0)  := mpi_ibsend(data, data.count, dtype, dest ,tag, y, ierr)
#$ header macro  (req),y.Irecv([data, dtype=data.dtype], source=ANY_SOURCE, tag=ANY_TAG) := mpi_irecv(data, data.count, dtype, source ,tag, y, req, ierr)


#$ header macro (x), y.Sendrecv(sendobj, dest, sendtag=0, recvbuf=x, source=ANY_SOURCE, recvtag=ANY_TAG, Stat=status) := mpi_sendrecv(sendobj, sendobj.count, sendobj.dtype,  dest, sendtag, recvbuf, recvbuf.count, recvbuf.dtype, source , recvtag, y, status, ierr)

#$ header macro y.Reduce(data, recvbuf, op=MPI_SUM, root=0) := mpi_reduce(data, recvbuf, data.count, data.dtype, op ,root, y, ierr)
#$ header macro y.Allreduce(data, recvbuf, op=MPI_SUM) := mpi_allreduce(data, recvbuf, data.count, data.dtype, op , y, ierr)
#$ header macro  x.Allgather(A,B) := mpi_allgather(A, A.count, A.dtype, B, B.count, B.dtype, x)


#$ header macro  y.Gather(data, recvbuf, root=0) := mpi_gather(data, data.count, data.dtype, recvbuf, recvbuf.count, recvbuf.dtype, root, y, ierr)

#$ header macro  y.Bcast(data, root=0) := mpi_bcast(data, data.count, data.dtype, root, y, ierr)

