from mpi4py import MPI
import numpy as np

from pyccel.epyccel import epyccel_mpi

#==============================================================================
# IMPORT MODULE TO BE TESTED, EPYCCELIZE IT, AND MAKE IT AVAILABLE TO ALL PROCS
#==============================================================================

def setup_module( module=None ):

    from modules import mpi_collective as pmod

    comm = MPI.COMM_WORLD
    fmod = epyccel_mpi( pmod, comm )

    if module:
        module.comm = comm
        module.pmod = pmod
        module.fmod = fmod
    else:
        globals().update( locals() )

#==============================================================================
# UNIT TESTS
#==============================================================================

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

#==============================================================================
# CLEAN UP GENERATED FILES AFTER RUNNING TESTS
#==============================================================================

def teardown_module():
    import os
    os.system( 'rm -f modules/__epyccel__*' )

#==============================================================================
# INTERACTIVE USAGE
#==============================================================================

if __name__ == '__main__':

    setup_module()

    test_np_allreduce()

    teardown_module()
