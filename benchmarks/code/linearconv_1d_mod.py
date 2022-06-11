# pythran export linearconv_1d(float[:], float[:], int, int, float, float, float)
def linearconv_1d(u: 'float[:]', un: 'float[:]',
                  nt: int, nx: int,
                  dt: float, dx: float, c: float):
    """ Solve the linear convection equation
    """

    for n in range(nt):
        un[:nx] = u[:nx]

        for i in range(1, nx):
            u[i] = un[i] - c * dt / dx * (un[i] - un[i-1])

