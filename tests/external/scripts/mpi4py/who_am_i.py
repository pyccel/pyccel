# pylint: disable=missing-function-docstring, missing-module-docstring
# coding: utf-8
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size_ = comm.Get_size()

print('I process ', rank, ', among ', size_, ' processes')
