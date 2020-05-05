# coding: utf-8

# TODO - improve communicate method
#      - debug reduce method: it must return the global error

from pyccel.stdlib.parallel.mpi import mpi_comm_size
from pyccel.stdlib.parallel.mpi import mpi_comm_rank
from pyccel.stdlib.parallel.mpi import mpi_comm_world
from pyccel.stdlib.parallel.mpi import mpi_status_size
from pyccel.stdlib.parallel.mpi import mpi_dims_create
from pyccel.stdlib.parallel.mpi import mpi_cart_create
from pyccel.stdlib.parallel.mpi import mpi_cart_coords
from pyccel.stdlib.parallel.mpi import mpi_cart_shift
from pyccel.stdlib.parallel.mpi import mpi_cart_sub
from pyccel.stdlib.parallel.mpi import mpi_comm_free
from pyccel.stdlib.parallel.mpi import mpi_type_contiguous
from pyccel.stdlib.parallel.mpi import mpi_type_vector
from pyccel.stdlib.parallel.mpi import mpi_type_commit
from pyccel.stdlib.parallel.mpi import mpi_type_free
from pyccel.stdlib.parallel.mpi import mpi_sendrecv
from pyccel.stdlib.parallel.mpi import mpi_allreduce
from pyccel.stdlib.parallel.mpi import MPI_DOUBLE
from pyccel.stdlib.parallel.mpi import MPI_SUM

#$ header class Cart(public)
#$ header method __init__(Cart, int [:], int [:], bool [:], bool)
#$ header method __del__(Cart)
#$ header method communicate(Cart, double [:,:])
#$ header method reduce(Cart, double)

class Cart(object):
    def __init__(self, npts, pads, periods, reorder):

        ntx = npts[0]
        nty = npts[1]

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

        # ...
        self.neighbour = zeros(self.n_neighbour, int)
        self.coords    = zeros(self.ndims, int)
        self.dims      = zeros(self.ndims, int)
        self.starts    = zeros(self.ndims, int)
        self.ends      = zeros(self.ndims, int)
        self.comm1d    = zeros(self.ndims, int)

        self.steps   = [1,1]
        self.pads    = pads
        self.periods = periods
        self.reorder = reorder
        # ...

        # ... TODO: remove from here
        ierr = -1
        size = -1
        self.rank = -1
        self.rank_in_topo = -1
        self.comm_cart = -1

        comm = mpi_comm_world
        mpi_comm_size(comm, size, ierr)
        mpi_comm_rank(comm, self.rank, ierr)
        # ...

        # ...
        # Know the number of processes along x and y
        mpi_dims_create (size, self.ndims, self.dims, ierr)
        # ...

        # ...
        # Create a 2d mpi cart
        mpi_cart_create (comm, self.ndims, self.dims, self.periods, self.reorder, self.comm_cart, ierr)

        # Know my coordinates in the topology
        mpi_comm_rank (self.comm_cart, self.rank_in_topo, ierr)
        mpi_cart_coords (self.comm_cart, self.rank_in_topo, self.ndims, self.coords, ierr)

        # X-axis limits
        sx = (self.coords[0]*ntx)/self.dims[0]
        ex = ((self.coords[0]+1)*ntx)/self.dims[0] - 1

        # Y-axis limits
        sy = (self.coords[1]*nty)/self.dims[1]
        ey = ((self.coords[1]+1)*nty)/self.dims[1] - 1
        # ...

        # ...
        self.starts[0] = sx
        self.ends[0]   = ex

        self.starts[1] = sy
        self.ends[1]   = ey
        # ...

        # ...
        self.sx = sx
        self.ex = ex + 1
        self.sy = sy
        self.ey = ey + 1
        # ...

        # ... grid without ghost cells
        self.r_x  = range(self.sx, self.ex, self.steps[0])
        self.r_y  = range(self.sy, self.ey, self.steps[1])

        self.indices = tensor (self.r_x, self.r_y)
        # ...

        # ...
        self.sx_ext = sx - self.pads[0]
        self.ex_ext = ex + self.pads[0] + 1
        self.sy_ext = sy - self.pads[1]
        self.ey_ext = ey + self.pads[1] + 1
        # ...

        # ... extended grid with ghost cells
        self.r_ext_x  = range(self.sx_ext, self.ex_ext, self.steps[0])
        self.r_ext_y  = range(self.sy_ext, self.ey_ext, self.steps[1])

        self.extended_indices = tensor (self.r_ext_x, self.r_ext_y)
        # ...

        # ... Neighbours
        #     Search of my West and East neigbours
        mpi_cart_shift (self.comm_cart, 0, self.pads[0], self.neighbour[west], self.neighbour[east], ierr)

        #     Search of my South and North neighbours
        mpi_cart_shift (self.comm_cart, 1, self.pads[1], self.neighbour[south], self.neighbour[north], ierr)
        # ...

        # ... Create 1d communicator within the cart
        flags = [True, False]
        mpi_cart_sub (self.comm_cart, flags, self.comm1d[0], ierr)

        flags = [False, True]
        mpi_cart_sub (self.comm_cart, flags, self.comm1d[1], ierr)
        # ...

        # ... Derived Types
        #     Creation of the type_line derived datatype to exchange points
        #     with northern to southern neighbours
        self.type_line = -1
        mpi_type_vector (ey-sy+1, 1, ex-sx+1+2*self.pads[0], MPI_DOUBLE, self.type_line, ierr)
        mpi_type_commit (self.type_line, ierr)

        #     Creation of the type_column derived datatype to exchange points
        #     with western to eastern neighbours
        self.type_column = -1
        mpi_type_contiguous (ex - sx + 1, MPI_DOUBLE, self.type_column, ierr)
        mpi_type_commit (self.type_column, ierr)
        # ...

    def __del__(self):
        ierr = -1

        # Free the datatype
        mpi_type_free (self.type_line, ierr)
        mpi_type_free (self.type_column, ierr)

        # Destruction of the communicators
        mpi_comm_free (self.comm_cart, ierr)

    def communicate(self, u):
        ierr = -1
        tag  = 1435
        status = zeros (mpi_status_size, int)

        # ... Constants
        north = 0
        east  = 1
        south = 2
        west  = 3
        # ...

        sx = self.starts[0]
        ex = self.ends[0]

        sy = self.starts[1]
        ey = self.ends[1]

        # ... Communication
        # Send to neighbour north and receive from neighbour south
        mpi_sendrecv (  u[sx, sy], 1, self.type_line, self.neighbour[north], tag, u[ex+1, sy], 1, self.type_line, self.neighbour[south], tag, self.comm_cart, status, ierr)

        # Send to neighbour south and receive from neighbour north
        mpi_sendrecv (  u[ex, sy], 1, self.type_line, self.neighbour[south], tag, u[sx-1, sy], 1, self.type_line, self.neighbour[north], tag, self.comm_cart, status, ierr)

        # Send to neighbour west  and receive from neighbour east
        mpi_sendrecv (  u[sx, sy], 1, self.type_column, self.neighbour[west], tag, u[sx, ey+1], 1, self.type_column, self.neighbour[east], tag, self.comm_cart, status, ierr)

        # Send to neighbour east  and receive from neighbour west
        mpi_sendrecv (  u[sx, ey], 1, self.type_column, self.neighbour[east], tag, u[sx, sy-1], 1, self.type_column, self.neighbour[west], tag, self.comm_cart, status, ierr)
        # ...

    def reduce(self, x):
        ierr     = -1

        global_x = 0.0

        mpi_allreduce (x, global_x, 1, MPI_DOUBLE, MPI_SUM, self.comm_cart, ierr)
        print(global_x, x)
