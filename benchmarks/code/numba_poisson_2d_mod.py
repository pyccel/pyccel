from numba import njit

@njit(fastmath=True)
def poisson_2d(p: 'float[:,:]', pd: 'float[:,:]', b: 'float[:,:]',
               nx: int, ny: int, nt: int, dx: float, dy: float):

    row, col = p.shape
    # Source
    b[ny // 4, nx // 4]  = 100
    b[3 * ny // 4, 3 * nx // 4] = -100


    for it in range(nt):
        pd[:,:] = p[:,:]

        for j in range(2, row):
            for i in range(2, col):
                p[j-1, i-1] = (((pd[j-1, i] + pd[j-1, i-2]) * dy**2 +
                                (pd[j, i-1] + pd[j-2, i-1]) * dx**2 -
                                b[j-1, i-1] * dx**2 * dy**2) /
                                (2 * (dx**2 + dy**2)))
        p[0, :] = 0
        p[ny-1, :] = 0
        p[:, 0] = 0
        p[:, nx-1] = 0

