# coding: utf-8

from pyccel.stdlib.parallel.mpi import mpi_init
from pyccel.stdlib.parallel.mpi import mpi_finalize
from pyccel.stdlib.parallel.mpi import mpi_comm_size
from pyccel.stdlib.parallel.mpi import mpi_comm_rank
from pyccel.stdlib.parallel.mpi import mpi_comm_world
from pyccel.stdlib.parallel.mpi import mpi_proc_null
from pyccel.stdlib.parallel.mpi import mpi_status_size
from pyccel.stdlib.parallel.mpi import mpi_sendrecv
from pyccel.stdlib.parallel.mpi import MPI_DOUBLE

# we need to declare these variables somehow,
# since we are calling mpi subroutines
ierr    = -1
comsize = -1
rank    = -1

mpi_init(ierr)

comm = mpi_comm_world

mpi_comm_size(comm, comsize, ierr)
mpi_comm_rank(comm, rank, ierr)

# ...
xleft  = -12.0
xright =  12.0

totpoints = 100
kappa     = 1.0
nsteps    = 10

ao        = 1.0
sigmao    = 1.0
time      = 0.0

old       = 1
new       = 2

lefttag   = 1
righttag  = 2

locnpoints = totpoints / comsize
# ...

# ...
startn = rank*locnpoints + 1
endn   = startn + locnpoints
if rank == comsize-1:
    endn = totpoints+1
locnpoints = endn-startn

left = rank-1
if left < 0:
    left = mpi_proc_null

right = rank+1
if right >= comsize:
    right = mpi_proc_null

dx = (xright-xleft)/(totpoints-1)
dt = dx**2 * kappa/10.0

locxleft = xleft + dx*(startn-1)
# ...

# ...
# allocate data, including ghost cells: old and new timestep
# theory doesn't need ghost cells, but we include it for simplicity

n_points = locnpoints+2

temperature = zeros((n_points,2), double)
theory      = zeros(n_points, double)
x           = zeros(n_points, double)
xx          = zeros(n_points, double)

#setup initial conditions

time = 0.0
for i in range(1,locnpoints+2):
    x[i] = locxleft + (i-1)*dx

xx = x**2 / (2.0 * sigmao**2)
temperature[:,old] = ao*exp(-xx)
theory  = ao*exp(-xx)

xxl = (xleft  - dx)**2 / (2.0 *sigmao**2)
xxr = (xright + dx)**2 / (2.0 *sigmao**2)

fixedlefttemp  = ao*exp(-xxl)
fixedrighttemp = ao*exp(-xxr)
# ...

# ...
# evolve
# step-1: boundary conditions: keep endpoint temperatures fixed.

status = zeros(mpi_status_size, int)

for step in range(0, nsteps):
    temperature[1,old] = fixedlefttemp
    temperature[locnpoints+2,old] = fixedrighttemp

    #exchange boundary information
    bc = temperature[locnpoints+1,old]
    y  = 0.0
    mpi_sendrecv (bc, 1, MPI_DOUBLE, right, righttag,
                  y,  1, MPI_DOUBLE, left, righttag,
                  comm, status, ierr)

    temperature[1,old] = y

    bc = temperature[2,old]
    y  = 0.0
    mpi_sendrecv (bc, 1, MPI_DOUBLE, left, lefttag,
                  y,  1, MPI_DOUBLE, right, lefttag,
                  comm, status, ierr)

    temperature[locnpoints+2,old] = y

    #update solution
    for i in range(2,locnpoints+1):
        temperature[i,new] = temperature[i,old] + dt*kappa/dx**2 * (temperature[i+1,old] - 2*temperature[i,old] + temperature[i-1,old])

    time = time + dt

    #update correct solution
    s = 2.0 * kappa * time + sigmao**2
    sigma  = sqrt(s)
    a      = ao*sigmao/sigma
    xx     = x**2 / (2.0*sigma**2)
    theory = a*exp(-xx)

    old = new
    new = new + 1
    if new > 2:
        new = 1

mpi_finalize(ierr)
