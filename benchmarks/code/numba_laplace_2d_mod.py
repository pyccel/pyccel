from numba import njit
import numpy as np

@njit(fastmath=True)
def laplace_2d(p: 'float[:,:]', y: 'float[:]',
               dx: float, dy: float, l1norm_target: float):
    """ Solve the Laplace equation
    """

    row, col = p.shape
    pn = np.empty((row,col))

    l1norm = 1.
    while l1norm > l1norm_target:
        pn[:,:] = p[:,:]

        p[1:-1, 1:-1] = ((dy**2 * (pn[1:-1, 2:] + pn[1:-1, 0:-2]) +
                         dx**2 * (pn[2:, 1:-1] + pn[0:-2, 1:-1])) /
                        (2 * (dx**2 + dy**2)))

        p[:, 0] = 0  # p = 0 @ x = 0
        p[:, -1] = y  # p = y @ x = 2
        p[0, :] = p[1, :]  # dp/dy = 0 @ y = 0
        p[-1, :] = p[-2, :]  # dp/dy = 0 @ y = 1
        l1norm = np.sum(np.abs(p[:]) - np.abs(pn[:])) / np.sum(np.abs(pn[:]))

