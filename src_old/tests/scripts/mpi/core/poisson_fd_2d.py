# coding: utf-8

from pyccel.stdlib.parallel.mpi import mpi_init
from pyccel.stdlib.parallel.mpi import mpi_finalize
from pyccel.stdlib.parallel.mpi import mpi_comm_size
from pyccel.stdlib.parallel.mpi import mpi_comm_rank
from pyccel.stdlib.parallel.mpi import mpi_comm_world
from pyccel.stdlib.parallel.mpi import mpi_status_size
from pyccel.stdlib.parallel.mpi import MPI_INTEGER
from pyccel.stdlib.parallel.mpi import MPI_DOUBLE
from pyccel.stdlib.parallel.mpi import MPI_SUM
from pyccel.stdlib.parallel.mpi import mpi_sendrecv
from pyccel.stdlib.parallel.mpi import mpi_dims_create
from pyccel.stdlib.parallel.mpi import mpi_cart_create
from pyccel.stdlib.parallel.mpi import mpi_cart_coords
from pyccel.stdlib.parallel.mpi import mpi_cart_shift
from pyccel.stdlib.parallel.mpi import mpi_comm_free
from pyccel.stdlib.parallel.mpi import mpi_type_contiguous
from pyccel.stdlib.parallel.mpi import mpi_type_vector
from pyccel.stdlib.parallel.mpi import mpi_type_commit
from pyccel.stdlib.parallel.mpi import mpi_type_free
from pyccel.stdlib.parallel.mpi import mpi_allreduce

# ...
ntx = 16
nty = 16

# Grid spacing
hx = 1.0/(ntx+1)
hy = 1.0/(nty+1)

# Equation Coefficients
c0 = (0.5*hx*hx*hy*hy)/(hx*hx+hy*hy)
c1 = 1.0/(hx*hx)
c2 = 1.0/(hy*hy)
# ...

# we need to declare these variables somehow,
# since we are calling mpi subroutines
ierr = -1
size = -1

# rank in comm worl
rank = -1

# rank is 2d cart
rank_in_topo = -1

# 2d cart communicator
comm_2d = -1

mpi_init(ierr)

comm = mpi_comm_world
mpi_comm_size(comm, size, ierr)
mpi_comm_rank(comm, rank, ierr)

north = 0
east  = 1
south = 2
west  = 3

ndims   = 2
steps   = [1, 1]
periods = [False, True]

reorder = False

neighbour = zeros(4, int)
coords    = zeros(2, int)
dims      = zeros(ndims, int)

# Know the number of processes along x and y
mpi_dims_create (size, ndims, dims, ierr)

# ...
# Create a 2d mpi cart
mpi_cart_create (comm, ndims, dims, periods, reorder, comm_2d, ierr)

# Know my coordinates in the topology
mpi_comm_rank (comm_2d, rank_in_topo, ierr)
mpi_cart_coords (comm_2d, rank_in_topo, ndims, coords, ierr)

# X-axis limits
sx = (coords[0]*ntx)/dims[0]
ex = ((coords[0]+1)*ntx)/dims[0] - 1

# Y-axis limits
sy = (coords[1]*nty)/dims[1]
ey = ((coords[1]+1)*nty)/dims[1] - 1

# ... Neighbours
#     Search of my West and East neigbours
mpi_cart_shift (comm_2d, 0, steps[0], neighbour[west], neighbour[east], ierr)

#     Search of my South and North neighbours
mpi_cart_shift (comm_2d, 1, steps[1], neighbour[south], neighbour[north], ierr)
# ...

# ... Derived Types
#     Creation of the type_line derived datatype to exchange points
#     with northern to southern neighbours
type_line = -1
mpi_type_vector (ey-sy+1, 1, ex-sx+3, MPI_DOUBLE, type_line, ierr)
mpi_type_commit (type_line, ierr)

#     Creation of the type_column derived datatype to exchange points
#     with western to eastern neighbours
type_column = -1
mpi_type_contiguous (ex - sx + 1, MPI_DOUBLE, type_column, ierr)
mpi_type_commit (type_column, ierr)
# ...

# ...
u       = vector((sx-1,sy-1), (ex+1, ey+1))
u_new   = vector((sx-1,sy-1), (ex+1, ey+1))
u_exact = vector((sx-1,sy-1), (ex+1, ey+1))
f       = vector((sx-1,sy-1), (ex+1, ey+1))

# Initialization
x = 0.0
y = 0.0
for i in range(sx, ex+1):
    for j in range(sy, ey+1):
        x = i*hx
        y = j*hy
#        print('> rank : ',rank_in_topo, '(i,j) = ',i,j)

        f[i, j] = 2.0*(x*x-x+y*y-y)
        u_exact[i, j] = x*y*(x-1.0)*(y-1.0)
# ...

# Linear solver tolerance
tol = 1.0e-10

tag = 1234
status = zeros (mpi_status_size, int)

n_iterations = 1000
for it in range(0, n_iterations):
    u[sx:ex+1,sy:ey+1] = u_new[sx:ex+1,sy:ey+1]

    # ... Communication
    # Send to neighbour north and receive from neighbour south
    mpi_sendrecv (  u[sx, sy], 1, type_line, neighbour[north], tag, u[ex+1, sy], 1, type_line, neighbour[south], tag, comm_2d, status, ierr)

    # Send to neighbour south and receive from neighbour north
    mpi_sendrecv (  u[ex, sy], 1, type_line, neighbour[south], tag, u[sx-1, sy], 1, type_line, neighbour[north], tag, comm_2d, status, ierr)

    # Send to neighbour west  and receive from neighbour east
    mpi_sendrecv (  u[sx, sy], 1, type_column, neighbour[west], tag, u[sx, ey+1], 1, type_column, neighbour[east], tag, comm_2d, status, ierr)

    # Send to neighbour east  and receive from neighbour west
    mpi_sendrecv (  u[sx, ey], 1, type_column, neighbour[east], tag, u[sx, sy-1], 1, type_column, neighbour[west], tag, comm_2d, status, ierr)
    # ...

    # ... Computation of u at the n+1 iteration
    for i in range(sx, ex+1):
        for j in range(sy, ey+1):
            u_new[i, j] = c0 * (c1*(u[i+1, j] + u[i-1, j]) + c2*(u[i, j+1] + u[i, j-1]) - f[i, j])
    # ...

    # ... Computation of the global error
    u_error = 0.0
    for i in range(sx, ex+1):
        for j in range(sy, ey+1):
            u_error += abs(u[i,j]-u_new[i,j])
    local_error = u_error/(ntx*nty)

    # Reduction
    global_error = 0.0
    mpi_allreduce (local_error, global_error, 1, MPI_DOUBLE, MPI_SUM, comm_2d, ierr)
    # ...

    # ...
    if (global_error < tol) or (it == n_iterations - 1):
        if rank == 0:
            print ("> convergence after ", it, " iterations")
            print ("  local  error = ", local_error)
            print ("  global error = ", global_error)
        break
    # ...

# Free the datatype
mpi_type_free (type_line, ierr)
mpi_type_free (type_column, ierr)

# Destruction of the communicators
mpi_comm_free (comm_2d, ierr)

mpi_finalize(ierr)
