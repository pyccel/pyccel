# pylint: disable=missing-function-docstring, missing-module-docstring
from mpi4py import MPI

if __name__ == '__main__':
    # we need to declare these variables somehow,
    # since we are calling mpi subroutines

    size = -1
    rank = -1
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    nx = 4
    ny = 3 * 2
    x = [0., 0., 0., 0.]
    y = [[0., 0.], [0., 0.], [0., 0.]]

    if rank == 0:
        x[:] = 1.0
        y[:,:] = 1.0

    source = 0
    dest   = 1


    # ...
    tag1 = 1234
    if rank == source:
        comm.send(x,dest, tag=tag1)
        print("> test 1: processor ", rank, " sent ", x)

    if rank == dest:
        x=comm.recv(source, tag1)
        print("> test 1: processor ", rank, " got  ", x)
    # ...

    # ...
    tag2 = 5678
    if rank == source:
        x[:] = 0.0
        x[1] = 2.0
        comm.send(x[1],dest, tag2)
        print("> test 2: processor ", rank, " sent ", x[1])

    if rank == dest:
        x[1]=comm.recv(source, tag2)
        print("> test 2: processor ", rank, " got  ", x[1])
    # ...

    # ...
    tag3 = 4321
    if rank == source:
        comm.send(y,dest, tag3)
        print("> test 3: processor ", rank, " sent ", y)

    if rank == dest:
        y=comm.recv(source, tag3)
        print("> test 3: processor ", rank, " got  ", y)
    # ...

    # ...
    tag4 = 8765
    if rank == source:
        y[:,:] = 0.0
        y[1,1] = 2.0
        comm.send(y[1,1],dest, tag4)
        print("> test 4: processor ", rank, " sent ", y[1,1])

    if rank == dest:
        y[1,1]=comm.recv(source, tag4)
        print("> test 4: processor ", rank, " got  ", y[1,1])
    # ...

    # ...
    tag5 = 6587
    if rank == source:
        comm.send(y[1,:],dest, tag5)
        print("> test 5: processor ", rank, " sent ", y[1,:])

    if rank == dest:
        y[1,:]=comm.recv(source, tag5)
        print("> test 5: processor ", rank, " got  ", y[1,:])
    # ...
