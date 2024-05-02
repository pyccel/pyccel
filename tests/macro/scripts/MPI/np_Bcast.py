# pylint: disable=missing-function-docstring, missing-module-docstring
from mpi4py import MPI
from numpy import zeros

if __name__ == '__main__':
    size = -1
    rank_in_world = -1

    comm = MPI.COMM_WORLD
    rank_in_world = comm.Get_rank()

    size = comm.Get_size()
    master = 0
    m      = 8

    a = zeros(m, 'int')

    if rank_in_world == 1:
        a[:] = 1
    if rank_in_world == 2:
        a[:] = 2

    key = rank_in_world
    if rank_in_world == 1:
        key = -1
    if rank_in_world == 2:
        key = -1

    two   = 2
    color = rank_in_world % two

    newcomm = -1

    newcomm = comm.Split(color, key)

    # Broadcast of the message by the rank process master of
    # each communicator to the processes of its group
    newcomm.Bcast(a, master)

    print("> processor ", rank_in_world, " has a = ", a)

    # Destruction of the communicators
    newcomm.Free()


