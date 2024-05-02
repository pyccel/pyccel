# coding: utf-8

from pyccel.mpi import *

ierr = mpi_init()

comm = mpi_comm_world
size = comm.size
rank = comm.rank

n = 4
x = zeros(n, double)
y = zeros((3,2), double)

if rank == 0:
    x = 1.0
    y = 1.0

source = 0
dest   = 1
tagx = 1234
if rank == source:
    ierr = comm.send(x, dest, tagx)
    print(("processor ", rank, " sent ", x))

if rank == dest:
    ierr = comm.recv(x, source, tagx)
    print(("processor ", rank, " got  ", x))

tag1 = 5678
if rank == source:
    x[1] = 2.0
    ierr = comm.send(x[1], dest, tag1)
    print(("processor ", rank, " sent x(1) = ", x[1]))

if rank == dest:
    ierr = comm.recv(x[1], source, tag1)
    print(("processor ", rank, " got  x(1) = ", x[1]))


tagx = 4321
if rank == source:
    ierr = comm.send(y, dest, tagx)
    print(("processor ", rank, " sent ", y))

if rank == dest:
    ierr = comm.recv(y, source, tagx)
    print(("processor ", rank, " got  ", y))

tag1 = 8765
if rank == source:
    y[1,1] = 2.0
    ierr = comm.send(y[1,1], dest, tag1)
    print(("processor ", rank, " sent y(1,1) = ", y[1,1]))

if rank == dest:
    ierr = comm.recv(y[1,1], source, tag1)
    print(("processor ", rank, " got  y(1,1) = ", y[1,1]))

tag1 = 6587
if rank == source:
    y[1,:] = 2.0
    ierr = comm.send(y[1,:], dest, tag1)
    print(("processor ", rank, " sent y(1,:) = ", y[1,:]))

if rank == dest:
    ierr = comm.recv(y[1,:], source, tag1)
    print(("processor ", rank, " got  y(1,:) = ", y[1,:]))


ierr = mpi_finalize()
