#$ header metavar module_name='mpi4py'
#$ header metavar module_version='3.1'
#$ header metavar ignore_at_import=True
#$ header metavar import_all=True

from pyccel.stdlib.internal.mpi import mpi_comm_world

from pyccel.stdlib.internal.mpi import mpi_send
from pyccel.stdlib.internal.mpi import mpi_ssend
from pyccel.stdlib.internal.mpi import mpi_bsend

from pyccel.stdlib.internal.mpi import mpi_isend
from pyccel.stdlib.internal.mpi import mpi_issend
from pyccel.stdlib.internal.mpi import mpi_ibsend

from pyccel.stdlib.internal.mpi import mpi_recv
from pyccel.stdlib.internal.mpi import mpi_irecv

from pyccel.stdlib.internal.mpi import mpi_sendrecv

from pyccel.stdlib.internal.mpi import mpi_bcast

from pyccel.stdlib.internal.mpi import mpi_barrier
from pyccel.stdlib.internal.mpi import mpi_gather
from pyccel.stdlib.internal.mpi import mpi_allgatherv

from pyccel.stdlib.internal.mpi import mpi_reduce
from pyccel.stdlib.internal.mpi import mpi_allreduce

from pyccel.stdlib.internal.mpi import mpi_waitall

from pyccel.stdlib.internal.mpi import mpi_comm_split
from pyccel.stdlib.internal.mpi import mpi_comm_free

from pyccel.stdlib.internal.mpi import mpi_type_vector
from pyccel.stdlib.internal.mpi import mpi_type_commit
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

from pyccel.stdlib.internal.mpi import MPI_LOGICAL
from pyccel.stdlib.internal.mpi import MPI_INTEGER
from pyccel.stdlib.internal.mpi import MPI_INTEGER8
from pyccel.stdlib.internal.mpi import MPI_REAL4
from pyccel.stdlib.internal.mpi import MPI_REAL8
from pyccel.stdlib.internal.mpi import MPI_COMPLEX8
from pyccel.stdlib.internal.mpi import MPI_COMPLEX16
from pyccel.stdlib.internal.mpi import MPI_CHARACTER

from pyccel.stdlib.internal.mpiext import mpiext_get_rank
from pyccel.stdlib.internal.mpiext import mpiext_get_size

#===================================================================================

#$ header class MPI_(public)
#$ header method __init__(MPI_)

class MPI_:
    def __init__(self):
        self.COMM_WORLD = -1
        self.Request    = -1

        self.SUM        = MPI_SUM
        self.PROD       = MPI_PROD
        self.MAX        = MPI_MAX
        self.MIN        = MPI_MIN
        self.MAXLOC     = MPI_MAXLOC
        self.MINLOC     = MPI_MINLOC
        self.LAND       = MPI_LAND
        self.LOR        = MPI_LOR
        self.LXOR       = MPI_LXOR

        self.LOGICAL    = MPI_LOGICAL
        self.INT        = MPI_INTEGER
        self.INTEGER    = MPI_INTEGER
        self.INTEGER8   = MPI_INTEGER8
        self.FLOAT      = MPI_REAL4
        self.DOUBLE     = MPI_REAL8
        self.COMPLEX    = MPI_COMPLEX8
        self.COMPLEX16  = MPI_COMPLEX16
        self.CHAR       = MPI_CHARACTER
        self.CHARACTER  = MPI_CHARACTER

MPI = MPI_()

#====================================================================================

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

#$ header macro x.LOGICAL    := MPI_LOGICAL
#$ header macro x.INT        := MPI_INTEGER
#$ header macro x.INTEGER    := MPI_INTEGER
#$ header macro x.FLOAT      := MPI_REAL4
#$ header macro x.DOUBLE     := MPI_REAL8
#$ header macro x.COMPLEX    := MPI_COMPLEX8
#$ header macro x.COMPLEX16  := MPI_COMPLEX16
#$ header macro x.CHAR       := MPI_CHARACTER
#$ header macro x.CHARACTER  := MPI_CHARACTER

#====================================================================================

#$ header macro y.Get_rank() := mpiext_get_rank(y)
#$ header macro y.Get_size() := mpiext_get_size(y)

#$ header macro y.rank := mpiext_get_rank(y)
#$ header macro y.size := mpiext_get_size(y)

#====================================================================================

#......................
#lower-case letter functions
#......................

ierr = -1

#$ header macro  y.send (data, dest, tag=0) := mpi_send (data, data.count, data.dtype, int32(dest), int32(tag), y, ierr)
#$ header macro  y.ssend(data, dest, tag=0) := mpi_ssend(data, data.count, data.dtype, int32(dest), int32(tag), y, ierr)
#$ header macro  y.bsend(data, dest, tag=0) := mpi_bsend(data, data.count, data.dtype, int32(dest), int32(tag), y, ierr)

#$ header macro (req), y.isend (data, dest, tag=0) := mpi_isend (data, data.count, data.dtype, int32(dest), int32(tag), y, req, ierr)
#$ header macro (req), y.issend(data, dest, tag=0) := mpi_issend(data, data.count, data.dtype, int32(dest), int32(tag), y, req, ierr)
#$ header macro (req), y.ibsend(data, dest, tag=0) := mpi_ibsend(data, data.count, data.dtype, int32(dest), int32(tag), y, req, ierr)

#$ header macro (x), y.recv(source=0, tag=0) := mpi_recv(x, x.count, x.dtype, int32(source), int32(tag), y, MPI_STATUS_IGNORE, ierr)

#$ header macro (x), y.sendrecv(sendobj, dest, sendtag=0, source=ANY_SOURCE, recvtag=ANY_TAG) := mpi_sendrecv(sendobj, sendobj.count, sendobj.dtype, int32(dest), int32(sendtag), x, x.count, x.dtype, int32(source), int32(recvtag), y, MPI_STATUS_IGNORE, ierr)

#$ header macro (x), y.reduce(data, op=MPI_SUM, root=0) := mpi_reduce(data, x, data.count, data.dtype, op, int32(root), y, ierr)
#$ header macro (x), y.allreduce(data, op=MPI_SUM) := mpi_allreduce(data, x, data.count, data.dtype, op, y, ierr)

#$ header macro       y.bcast(data, root=0)  := mpi_bcast(data, data.count, data.dtype, int32(root), y, ierr)
#$ header macro  (x), y.gather(data, root=0) := mpi_gather(data, data.count, data.dtype, x, data.count, x.dtype, int32(root), y, ierr)

#.....................
##$ header macro (x),y.scatter
##$ header macro (req), y.irecv
##$ header macro y.alltoall
#not_working for the moment
#.....................


#......................
#upper-case letter functions
#......................

#$ header macro (x), y.Split(color=0, key=0) := mpi_comm_split(y, int32(color), int32(key), x, ierr)
#$ header macro y.Free() := mpi_comm_free(y, ierr)
#$ header macro (datatype),y.Create_vector(count, blocklength, stride) := mpi_type_vector(int32(count), int32(blocklength), int32(stride), y.dtype, datatype, ierr)
#$ header macro x.Commit() := mpi_type_commit(x, ierr)


#$ header macro  y.Send([data, dtype=data.dtype], dest=0, tag=0)  := mpi_send(data, data.count, dtype, int32(dest), int32(tag), y, ierr)
#$ header macro  y.Recv([data, dtype=data.dtype], source=ANY_SOURCE, tag=ANY_TAG) := mpi_recv(data, data.count, data.dtype, int32(source), int32(tag), y, MPI_STATUS_IGNORE, ierr)

#$ header macro  (req), y.Isend ([data, count=data.count, dtype=data.dtype], dest=0, tag=0) := mpi_isend (data, int32(count), dtype, int32(dest), int32(tag), y, int32(req), ierr)
#$ header macro  (req), y.Issend([data, count=data.count, dtype=data.dtype], dest=0, tag=0) := mpi_issend(data, int32(count), dtype, int32(dest), int32(tag), y, ierr)
#$ header macro  (req), y.Ibsend([data, count=data.count, dtype=data.dtype], dest=0, tag=0) := mpi_ibsend(data, int32(count), dtype, int32(dest), int32(tag), y, ierr)
#$ header macro  (req), y.Irecv ([data, count=data.count, dtype=data.dtype], source=ANY_SOURCE, tag=ANY_TAG) := mpi_irecv(data, int32(count), dtype, int32(source), int32(tag), y, int32(req), ierr)


#$ header macro (x), y.Sendrecv(sendobj, dest, sendtag=0, recvbuf=x, source=ANY_SOURCE, recvtag=ANY_TAG) := mpi_sendrecv(sendobj, sendobj.count, sendobj.dtype, int32(dest), int32(sendtag), recvbuf, recvbuf.count, recvbuf.dtype, int32(source), int32(recvtag), y, MPI_STATUS_IGNORE, ierr)

#$ header macro y.Reduce(data, recvbuf, op=MPI_SUM, root=0) := mpi_reduce(data, recvbuf, data.count, data.dtype, op, int32(root), y, ierr)
#$ header macro y.Allreduce(data, recvbuf, op=MPI_SUM) := mpi_allreduce(data, recvbuf, data.count, data.dtype, op, y, ierr)
#$ header macro x.Allgatherv(A,[B,Bcounts,Bdisps,Bdtype = B.dtype]) := mpi_allgatherv(A, A.count, A.dtype, B, Bcounts, Bdisps, Bdtype, x, ierr)

#$ header macro  y.Gather(data, recvbuf, root=0) := mpi_gather(data, data.count, data.dtype, recvbuf, data.count, recvbuf.dtype, int32(root), y, ierr)

#$ header macro  y.Bcast(data, root=0) := mpi_bcast(data, data.count, data.dtype, int32(root), y, ierr)

#$ header macro  x.Waitall(req) := mpi_waitall(req.count, req, MPI_STATUSES_IGNORE, ierr)

#$ header macro  y.Barrier() := mpi_barrier(y, ierr)
