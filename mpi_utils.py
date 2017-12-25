# coding: utf-8

# TODO : steps, periods, reorder to be passed as arguments

from pyccel.stdlib.parallel.mpi import mpi_init
from pyccel.stdlib.parallel.mpi import mpi_finalize
from pyccel.stdlib.parallel.mpi import mpi_comm_size
from pyccel.stdlib.parallel.mpi import mpi_comm_rank
from pyccel.stdlib.parallel.mpi import mpi_comm_world
from pyccel.stdlib.parallel.mpi import mpi_status_size
from pyccel.stdlib.parallel.mpi import mpi_dims_create
from pyccel.stdlib.parallel.mpi import mpi_cart_create
from pyccel.stdlib.parallel.mpi import mpi_cart_coords
from pyccel.stdlib.parallel.mpi import mpi_cart_shift
from pyccel.stdlib.parallel.mpi import mpi_comm_free
from pyccel.stdlib.parallel.mpi import mpi_type_contiguous
from pyccel.stdlib.parallel.mpi import mpi_type_vector
from pyccel.stdlib.parallel.mpi import mpi_type_commit
from pyccel.stdlib.parallel.mpi import mpi_type_free

#$ header class Cart(public)
#$ header method __init__(Cart)
#$ header method __del__(Cart)

class Cart(object):
    def __init__(self):

        ntx = 16
        nty = 16
        steps   = [1, 1]
        periods = [False, True]
        reorder = False

        # ... TODO : to be computed using 'len'
        self.ndims       = 2
        self.n_neighbour = 4
        # ...

        # ... Constants
        north = 0
        east  = 1
        south = 2
        west  = 3
        # ...

        # ... TODO : use steps, periods, reorder arguments
        self.neighbour = zeros(self.n_neighbour, int)
        self.coords    = zeros(self.ndims, int)
        self.dims      = zeros(self.ndims, int)

        self.steps   = [1, 1]
        self.periods = [False, True]
        self.reorder = False
        # ...

        # ... TODO: remove from here
        ierr = -1
        size = -1
        rank = -1
        rank_in_topo = -1
        comm_2d = -1

        comm = mpi_comm_world
        mpi_comm_size(comm, size, ierr)
        mpi_comm_rank(comm, rank, ierr)
        # ...

        # ...
        # Know the number of processes along x and y
        mpi_dims_create (size, self.ndims, self.dims, ierr)
        # ...

        # ...
        # Create a 2d mpi cart
        mpi_cart_create (comm, self.ndims, self.dims, self.periods, self.reorder, comm_2d, ierr)

        # Know my coordinates in the topology
        mpi_comm_rank (comm_2d, rank_in_topo, ierr)
        mpi_cart_coords (comm_2d, rank_in_topo, self.ndims, self.coords, ierr)

        # X-axis limits
        sx = (self.coords[0]*ntx)/self.dims[0]
        ex = ((self.coords[0]+1)*ntx)/self.dims[0] - 1

        # Y-axis limits
        sy = (self.coords[1]*nty)/self.dims[1]
        ey = ((self.coords[1]+1)*nty)/self.dims[1] - 1
        # ...

        # ... Neighbours
        #     Search of my West and East neigbours
        mpi_cart_shift (comm_2d, 0, self.steps[0], self.neighbour[west], self.neighbour[east], ierr)

        #     Search of my South and North neighbours
        mpi_cart_shift (comm_2d, 1, self.steps[1], self.neighbour[south], self.neighbour[north], ierr)
        # ...

        # ... Derived Types
        #     Creation of the type_line derived datatype to exchange points
        #     with northern to southern neighbours
        self.type_line = -1
        mpi_type_vector (ey-sy+1, 1, ex-sx+3, MPI_DOUBLE, self.type_line, ierr)
        mpi_type_commit (self.type_line, ierr)

        #     Creation of the type_column derived datatype to exchange points
        #     with western to eastern neighbours
        self.type_column = -1
        mpi_type_contiguous (ex - sx + 1, MPI_DOUBLE, self.type_column, ierr)
        mpi_type_commit (self.type_column, ierr)
        # ...



        # ... TODO to be moved to __del__
        # Free the datatype
        mpi_type_free (self.type_line, ierr)
        mpi_type_free (self.type_column, ierr)

        # Destruction of the communicators
        mpi_comm_free (comm_2d, ierr)
        # ...

    def __del__(self):
        pass

ierr = -1

mpi_init(ierr)

p = Cart()

del p

mpi_finalize(ierr)
