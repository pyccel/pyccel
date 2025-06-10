"""
Pyccel header for MPI.
"""

# On travis, 'use mpi, only: mpi_allgather' is not working
# for this reason, we will ignore all imports 
# and tell pyccel to add 'use mpi' using import_all

#$ header metavar module_name='mpi'
#$ header metavar module_version='3.1'
#$ header metavar ignore_at_import=True
#$ header metavar import_all=True
#$ header metavar save=True
#$ header metavar external=False

from typing import Any
from numpy import int32

# ............................................................
#            MPI Constants
# ............................................................

mpi_comm_world : 'int32'
mpi_status_size : 'int32'
mpi_proc_null : 'int32'
MPI_LOGICAL : 'int32'
MPI_INTEGER : 'int32'
MPI_INTEGER8 : 'int32'
MPI_REAL4 : 'int32'
MPI_REAL8 : 'int32'
MPI_COMPLEX8 : 'int32'
MPI_COMPLEX16 : 'int32'
MPI_CHARACTER : 'int32'
MPI_SUM : 'int32'
MPI_PROD : 'int32'
MPI_MAX : 'int32'
MPI_MIN : 'int32'
MPI_MAXLOC : 'int32'
MPI_MINLOC : 'int32'
MPI_LAND : 'int32'
MPI_LOR : 'int32'
MPI_LXOR : 'int32'

# ............................................................
#
# ............................................................

def mpi_init(anon_0 : 'int32') -> None:
    ...

def mpi_finalize(anon_0 : 'int32') -> None:
    ...

def mpi_abort(anon_0 : 'int32', anon_1 : 'int32', anon_2 : 'int32') -> None:
    ...

# ............................................................
#          Communicator Accessors
# ............................................................
def mpi_comm_size(anon_0 : 'int32', anon_1 : 'int32', anon_2 : 'int32') -> None:
    ...

def mpi_comm_rank(anon_0 : 'int32', anon_1 : 'int32', anon_2 : 'int32') -> None:
    ...

# ............................................................
#          Point-to-Point Communication
# ............................................................
def mpi_recv(anon_0 : 'Any', anon_1 : 'int32', anon_2 : 'int32', anon_3 : 'int32', anon_4 : 'int32', anon_5 : 'int32', anon_6 : 'int32[:]', anon_7 : 'int32') -> None:
    ...

def mpi_send(anon_0 : 'Any', anon_1 : 'int32', anon_2 : 'int32', anon_3 : 'int32', anon_4 : 'int32', anon_5 : 'int32', anon_6 : 'int32') -> None:
    ...

def mpi_ssend(anon_0 : 'Any', anon_1 : 'int32', anon_2 : 'int32', anon_3 : 'int32', anon_4 : 'int32', anon_5 : 'int32', anon_6 : 'int32') -> None:
    ...

def mpi_bsend(anon_0 : 'Any', anon_1 : 'int32', anon_2 : 'int32', anon_3 : 'int32', anon_4 : 'int32', anon_5 : 'int32', anon_6 : 'int32') -> None:
    ...

def mpi_buffer_attach(anon_0 : 'Any', anon_1 : 'int32', anon_2 : 'int32') -> None:
    ...

def mpi_buffer_detach(anon_0 : 'Any', anon_1 : 'int32', anon_2 : 'int32') -> None:
    ...

def mpi_irecv(anon_0 : 'Any', anon_1 : 'int32', anon_2 : 'int32', anon_3 : 'int32', anon_4 : 'int32', anon_5 : 'int32', anon_6 : 'int32', anon_7 : 'int32') -> None:
    ...

def mpi_isend(anon_0 : 'Any', anon_1 : 'int32', anon_2 : 'int32', anon_3 : 'int32', anon_4 : 'int32', anon_5 : 'int32', anon_6 : 'int32', anon_7 : 'int32') -> None:
    ...

def mpi_issend(anon_0 : 'Any', anon_1 : 'int32', anon_2 : 'int32', anon_3 : 'int32', anon_4 : 'int32', anon_5 : 'int32', anon_6 : 'int32', anon_7 : 'int32') -> None:
    ...

def mpi_ibsend(anon_0 : 'Any', anon_1 : 'int32', anon_2 : 'int32', anon_3 : 'int32', anon_4 : 'int32', anon_5 : 'int32', anon_6 : 'int32', anon_7 : 'int32') -> None:
    ...

def mpi_sendrecv(anon_0 : 'Any', anon_1 : 'int32', anon_2 : 'int32', anon_3 : 'int32', anon_4 : 'int32', anon_5 : 'Any', anon_6 : 'int32', anon_7 : 'int32', anon_8 : 'int32', anon_9 : 'int32', anon_10 : 'int32', anon_11 : 'int32[:]', anon_12 : 'int32') -> None:
    ...

def mpi_sendrecv_replace(anon_0 : 'Any', anon_1 : 'int32', anon_2 : 'int32', anon_3 : 'int32', anon_4 : 'int32', anon_5 : 'int32', anon_6 : 'int32', anon_7 : 'int32', anon_8 : 'int32[:]', anon_9 : 'int32') -> None:
    ...

def mpi_barrier(anon_0 : 'int32', anon_1 : 'int32') -> None:
    ...

def mpi_bcast(anon_0 : 'Any', anon_1 : 'int32', anon_2 : 'int32', anon_3 : 'int32', anon_4 : 'int32', anon_5 : 'int32') -> None:
    ...

def mpi_scatter(anon_0 : 'Any', anon_1 : 'int32', anon_2 : 'int32', anon_3 : 'Any', anon_4 : 'int32', anon_5 : 'int32', anon_6 : 'int32', anon_7 : 'int32', anon_8 : 'int32') -> None:
    ...

def mpi_gather(anon_0 : 'Any', anon_1 : 'int32', anon_2 : 'int32', anon_3 : 'Any', anon_4 : 'int32', anon_5 : 'int32', anon_6 : 'int32', anon_7 : 'int32', anon_8 : 'int32') -> None:
    ...

def mpi_allgather(anon_0 : 'Any', anon_1 : 'int32', anon_2 : 'int32', anon_3 : 'Any', anon_4 : 'int32', anon_5 : 'int32', anon_6 : 'int32', anon_7 : 'int32') -> None:
    ...

def mpi_allgatherv(anon_0 : 'Any', anon_1 : 'int32', anon_2 : 'int32', anon_3 : 'Any', anon_4 : 'int32[:]', anon_5 : 'int32[:]', anon_6 : 'int32', anon_7 : 'int32') -> None:
    ...

def mpi_gatherv(anon_0 : 'Any', anon_1 : 'int32', anon_2 : 'int32', anon_3 : 'Any', anon_4 : 'int32[:]', anon_5 : 'int32[:]', anon_6 : 'int32', anon_7 : 'int32', anon_8 : 'int32', anon_9 : 'int32') -> None:
    ...

def mpi_alltoall(anon_0 : 'Any', anon_1 : 'int32', anon_2 : 'int32', anon_3 : 'Any', anon_4 : 'int32', anon_5 : 'int32', anon_6 : 'int32', anon_7 : 'int32') -> None:
    ...

def mpi_reduce(anon_0 : 'Any', anon_1 : 'Any', anon_2 : 'int32', anon_3 : 'int32', anon_4 : 'int32', anon_5 : 'int32', anon_6 : 'int32', anon_7 : 'int32') -> None:
    ...

def mpi_allreduce(anon_0 : 'Any', anon_1 : 'Any', anon_2 : 'int32', anon_3 : 'int32', anon_4 : 'int32', anon_5 : 'int32', anon_6 : 'int32') -> None:
    ...

def mpi_wait(anon_0 : 'int32', anon_1 : 'int32[:]', anon_2 : 'int32') -> None:
    ...

def mpi_waitall(anon_0 : 'int32', anon_1 : 'int32[:]', anon_2 : 'int32[:,:](order=C)', anon_3 : 'int32') -> None:
    ...

def mpi_waitany(anon_0 : 'int32', anon_1 : 'int32[:]', anon_2 : 'int32', anon_3 : 'int32[:]', anon_4 : 'int32') -> None:
    ...

def mpi_waitsome(anon_0 : 'int32', anon_1 : 'int32[:]', anon_2 : 'int32', anon_3 : 'int32[:]', anon_4 : 'int32[:,:](order=C)', anon_5 : 'int32') -> None:
    ...

def mpi_test(anon_0 : 'int32', anon_1 : 'bool', anon_2 : 'int32[:]', anon_3 : 'int32') -> None:
    ...

def mpi_testall(anon_0 : 'int32', anon_1 : 'bool', anon_2 : 'int32[:]', anon_3 : 'int32[:,:](order=C)', anon_4 : 'int32') -> None:
    ...

def mpi_testany(anon_0 : 'int32', anon_1 : 'int32[:]', anon_2 : 'int32', anon_3 : 'bool', anon_4 : 'int32[:]', anon_5 : 'int32') -> None:
    ...

def mpi_testsome(anon_0 : 'int32', anon_1 : 'int32[:]', anon_2 : 'int32', anon_3 : 'int32[:]', anon_4 : 'int32[:,:](order=C)', anon_5 : 'int32') -> None:
    ...

def mpi_comm_split(anon_0 : 'int32', anon_1 : 'int32', anon_2 : 'int32', anon_3 : 'int32', anon_4 : 'int32') -> None:
    ...

def mpi_comm_free(anon_0 : 'int32', anon_1 : 'int32') -> None:
    ...

def mpi_cart_create(anon_0 : 'int32', anon_1 : 'int32', anon_2 : 'int32[:]', anon_3 : 'int32[:]', anon_4 : 'bool', anon_5 : 'int32', anon_6 : 'int32') -> None:
    ...

def mpi_cart_coords(anon_0 : 'int32', anon_1 : 'int32', anon_2 : 'int32', anon_3 : 'int32[:]', anon_4 : 'int32') -> None:
    ...

def mpi_cart_shift(anon_0 : 'int32', anon_1 : 'int32', anon_2 : 'int32', anon_3 : 'int32', anon_4 : 'int32', anon_5 : 'int32') -> None:
    ...

def mpi_cart_sub(anon_0 : 'int32', anon_1 : 'int32[:]', anon_2 : 'int32', anon_3 : 'int32') -> None:
    ...

def mpi_dims_create(anon_0 : 'int32', anon_1 : 'int32', anon_2 : 'int32[:]', anon_3 : 'int32') -> None:
    ...

# ............................................................
#          Derived datatypes
# ............................................................
def mpi_type_contiguous(anon_0 : 'int32', anon_1 : 'int32', anon_2 : 'int32', anon_3 : 'int32') -> None:
    ...

def mpi_type_vector(anon_0 : 'int32', anon_1 : 'int32', anon_2 : 'int32', anon_3 : 'int32', anon_4 : 'int32', anon_5 : 'int32') -> None:
    ...

def mpi_type_commit(anon_0 : 'int32', anon_1 : 'int32') -> None:
    ...

def mpi_type_free(anon_0 : 'int32', anon_1 : 'int32') -> None:
    ...

def mpi_type_indexed(anon_0 : 'int32', anon_1 : 'int32[:]', anon_2 : 'int32[:]', anon_3 : 'int32', anon_4 : 'int32', anon_5 : 'int32') -> None:
    ...

def mpi_type_create_subarray(anon_0 : 'int32', anon_1 : 'int32[:]', anon_2 : 'int32[:]', anon_3 : 'int32[:]', anon_4 : 'int32', anon_5 : 'int32', anon_6 : 'int32', anon_7 : 'int32') -> None:
    ...

