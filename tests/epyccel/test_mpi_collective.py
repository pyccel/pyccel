# pylint: disable=missing-function-docstring, missing-module-docstring
from mpi4py import MPI
import numpy as np
import pytest

from pyccel.epyccel import epyccel
from modules        import mpi_collective as pmod

#==============================================================================
# IMPORT MODULE TO BE TESTED, EPYCCELIZE IT, AND MAKE IT AVAILABLE TO ALL PROCS
#==============================================================================

def setup_module( module=None ):

    comm = MPI.COMM_WORLD
    fmod = epyccel( pmod, comm=comm )

    if module:
        module.comm = comm
        module.fmod = fmod
    else:
        globals().update( locals() )

#==============================================================================
# UNIT TESTS
#==============================================================================
@pytest.mark.xfail(reason = 'issue 251: broken mpi4py support')
@pytest.mark.parallel
def test_np_allreduce( ne=15 ):
    """
    Initialize a 1D integer array with the process rank, and sum across
    all processes using an MPI_SUM global reduction operation.
    The exact result will be an integer array with all elements equal to

    N*(N-1)//2

    with 'N' the number of processes.

    Parameters
    ----------
    ne : int
        Size of 1D integer array.

    """
    # Send and receive buffers
    sendbuf     = np.ones ( ne, dtype='i' ) * comm.rank
    recvbuf_py  = np.empty( ne, dtype='i' )
    recvbuf_f90 = np.empty( ne, dtype='i' )

    # Exact value after MPI_SUM reduction operation on 'sendbuf'
    sz    = comm.size
    exact = sz*(sz-1)//2

    # Python
    pmod.np_allreduce( sendbuf, recvbuf_py )
    assert all( recvbuf_py == exact )

    # Fortran
    fmod.np_allreduce( sendbuf, recvbuf_f90 )
    assert np.array_equal( recvbuf_py, recvbuf_f90 )

# ...
@pytest.mark.xfail(reason = 'issue 251: broken mpi4py support')
@pytest.mark.parallel
def test_np_bcast( ne=15 ):

    root  = 0
    exact = np.arange( ne, dtype='i' )

    # Send/receive buffer
    if comm.rank == root:
        buf_py  = exact.copy()
        buf_f90 = exact.copy()
    else:
        buf_py  = np.zeros_like( exact )
        buf_f90 = np.zeros_like( exact )

    # Python
    pmod.np_bcast( buf_py, root )
    assert np.array_equal( buf_py, exact )

    # Fortran
    fmod.np_bcast( buf_f90, root )
    assert np.array_equal( buf_f90, exact )

# ...
@pytest.mark.xfail(reason = 'issue 251: broken mpi4py support')
@pytest.mark.parallel
def test_np_gather():

    root = 0
    nval = 10 * comm.size

    # Split array across all processes in communicator
    s = (nval *  comm.rank   ) // comm.size
    e = (nval * (comm.rank+1)) // comm.size

    # Send and receive buffers
    sendbuf = np.arange( s, e, dtype='i' )
    if comm.rank == root:
        recvbuf_py  = np.empty ( nval, dtype='i' )
        recvbuf_f90 = np.empty ( nval, dtype='i' )
        exact       = np.arange( nval, dtype='i' )
    else:
        recvbuf_py  = np.empty ( 0, dtype='i' )
        recvbuf_f90 = np.empty ( 0, dtype='i' )
        exact       = np.empty ( 0, dtype='i' )

    # Python
    pmod.np_gather( sendbuf, recvbuf_py, root )
    assert np.array_equal( recvbuf_py, exact )

    # Fortran
    fmod.np_gather( sendbuf, recvbuf_f90, root )
    assert np.array_equal( recvbuf_f90, exact )

#==============================================================================
# CLEAN UP GENERATED FILES AFTER RUNNING TESTS
#==============================================================================

def teardown_module():

    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        import os, glob
        dirname  = os.path.dirname( pmod.__file__ )
        pattern  = os.path.join( dirname, '__epyccel__*' )
        filelist = glob.glob( pattern )
        for f in filelist:
            os.remove( f )

    comm.Barrier()

#==============================================================================
# INTERACTIVE USAGE
#==============================================================================

if __name__ == '__main__':

    setup_module()

    test_np_allreduce()
    test_np_bcast()
    test_np_gather()

    teardown_module()
