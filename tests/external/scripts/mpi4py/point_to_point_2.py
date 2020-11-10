# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size_ = comm.Get_size()

n = 10
x = [0.]*n

if rank == 0:
    x[:] = 1.0

source = 0

if rank == source:
    x[1] = 2.0
    comm.send(x[1], 1, tag=1234)
    print("> processor ", rank, " sent x(1) = ", x)
# ...
