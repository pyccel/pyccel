from numba import njit

@njit(fastmath=True)
def nonlinearconv_1d(u: 'float[:]', un: 'float[:]',
                     nt: int, nx: int, dt: float, dx: float):
    """ Solve a non-linear convection equation
    """

    for n in range(nt):
        un[:] = u[:]
        for i in range(1, nx):
            u[i] = un[i] - un[i] * dt / dx * (un[i] - un[i-1])

