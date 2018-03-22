# coding: utf-8

from pyccel.mpi import *

ierr = mpi_init()

comm = mpi_comm_world
size = comm.size
rank = comm.rank

n = 4
x = zeros(n, double)
y = zeros(n, double)

if rank == 0:
    x = 1.0
    y = 2.0

tag0 = 1234
tag1 = 5678

prev = rank - 1
next = rank + 1
if rank == 0:
    prev = size - 1
if rank == size - 1:
    next = 0

req0 = 0
req1 = 0
req2 = 0
req3 = 0

ierr = comm.irecv(x, prev, tag0, req0)
ierr = comm.irecv(y, next, tag1, req1)
ierr = comm.isend(x, prev, tag1, req2)
ierr = comm.isend(y, next, tag0, req3)

status_size = mpi_status_size
reqs  = zeros(4, int)
stats = zeros((status_size,4), int)

reqs[0] = req0
reqs[1] = req1
reqs[2] = req2
reqs[3] = req3

ierr = mpi_waitall(reqs, stats)

ierr = mpi_finalize()
