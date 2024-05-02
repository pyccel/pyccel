# pylint: disable=missing-function-docstring, missing-module-docstring
from mpi4py import MPI
import numpy as np
import pytest

from pyccel.epyccel import epyccel
from modules        import mpi_point_to_point as pmod

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
def test_np_sendrecv():

    rank = comm.Get_rank()
    size = comm.Get_size()

    # Send messages around in a ring
    source = (rank - 1) % size
    dest   = (rank + 1) % size

    # Create message to be sent, initialize receive buffer, choose some tag
    msg = rank + 1000
    tag = 1234

    sendbuf     = np.array( [msg], dtype='i' )
    recvbuf_py  = np.empty_like( sendbuf )
    recvbuf_f90 = np.empty_like( sendbuf )

    # Python
    pmod.np_sendrecv( sendbuf, dest, tag, recvbuf_py , source, tag )

    # Fortran
    fmod.np_sendrecv( sendbuf, dest, tag, recvbuf_f90, source, tag )

    assert np.array_equal( recvbuf_py, recvbuf_f90 )

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
