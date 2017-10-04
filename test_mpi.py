# coding: utf-8

ierr = mpi_init()

comm    = mpi_comm_world
comsize = comm.size
rank    = comm.rank

xleft  = -12.0
xright =  12.0

totpoints = 100
kappa     = 1.0
nsteps    = 10

locnpoints = totpoints/comsize

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

ierr = mpi_finalize()
