# coding: utf-8

from pyccel.mpi import *

ierr = mpi_init()

comm      = mpi_comm_world
comsize   = comm.size
rank      = comm.rank

xleft     = -12.0
xright    =  12.0

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

#allocate data, including ghost cells: old and new timestep
#theory doesn't need ghost cells, but we include it for simplicity

temperature = zeros((locnpoints+2,2), double)
theory      = zeros(locnpoints+2, double)
x           = zeros(locnpoints+2, double)
xx          = zeros(locnpoints+2, double)

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

#evolve
#step-1: boundary conditions: keep endpoint temperatures fixed.
for step in range(0, nsteps):
    temperature[1,old] = fixedlefttemp
    temperature[locnpoints+2,old] = fixedrighttemp

    #exchange boundary information
    ierr = comm.sendrecv(temperature[locnpoints+1,old], right, righttag, temperature[1,old], left, righttag)
    ierr = comm.sendrecv(temperature[2,old], left, lefttag, temperature[locnpoints+2,old], right, lefttag)

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

ierr = mpi_finalize()
