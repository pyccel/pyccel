import numpy as np

#==============================================================================
# pythran export linearconv_1d(float[:], float[:], int, int, float, float, float)
def linearconv_1d(u: 'float[:]', un: 'float[:]',
                  nt: int, nx: int,
                  dt: float, dx: float, c: float):

    for n in range(nt):
        un[:nx] = u[:nx]

        for i in range(1, nx):
            u[i] = un[i] - c * dt / dx * (un[i] - un[i-1])

#==============================================================================
# pythran export lineardiff_1d(float[:], float[:], int, int, float, float, float)
def lineardiff_1d(u: 'float[:]', un: 'float[:]',
                  nt: int, nx: int,
                  dt: float, dx: float, nu: float):

    for n in range(nt):
        un[:] = u[:]
        for i in range(1, nx - 1):
            u[i] = un[i] + nu * dt / dx**2 * (un[i+1] - 2 * un[i] + un[i-1])

#==============================================================================
# pythran export nonlinearconv_1d(float[:], float[:], int, int, float, float)
def nonlinearconv_1d(u: 'float[:]', un: 'float[:]',
                     nt: int, nx: int, dt: float, dx: float):

    for n in range(nt):
        un[:] = u[:]
        for i in range(1, nx):
            u[i] = un[i] - un[i] * dt / dx * (un[i] - un[i-1])

#==============================================================================
# pythran export burgers_1d(float[:], float[:], int, int, float, float, float)
def burgers_1d(u: 'float[:]', un: 'float[:]',
               nt: int, nx: int,
               dt: float, dx: float, nu: float):

    for n in range(nt):
        un[:] = u[:]
        for i in range(1, nx-1):
            u[i] = un[i] - un[i] * dt / dx *(un[i] - un[i-1]) + nu * dt / dx**2 * \
                    (un[i+1] - 2 * un[i] + un[i-1])

        u[0] = un[0] - un[0] * dt / dx * (un[0] - un[-2]) + nu * dt / dx**2 * \
                    (un[1] - 2 * un[0] + un[-2])
        u[-1] = u[0]

#==============================================================================
# pythran export linearconv_2d(float[:,:], float[:,:], int, float, float, float, float)
def linearconv_2d(u: 'float[:,:]', un: 'float[:,:]',
                  nt: int,  dt: float, dx: float, dy: float, c: float):

    row, col = u.shape

    for n in range(nt + 1): ##loop across number of time steps
        un[:,:] = u[:,:]

        for j in range(1, row):
            for i in range(1, col):
                u[j, i] = (un[j, i] - (c * dt / dx * (un[j, i] - un[j, i - 1])) -
                                      (c * dt / dy * (un[j, i] - un[j - 1, i])))
                u[0, :] = 1
                u[-1, :] = 1
                u[:, 0] = 1
                u[:, -1] = 1

#==============================================================================
# pythran export lineardiff_2d(float[:,:], float[:,:], int, float, float, float, float)
def lineardiff_2d(u: 'float[:,:]', un: 'float[:,:]',
                  nt: int, dt: float, dx: float, dy: float, nu: float):
    row, col = u.shape

    ##Assign initial conditions
    #set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
    u[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2


    for n in range(nt + 1):
        un[:,:] = u[:,:]

        for j in range(2, row):
            for i in range(2, col):
                u[j-1, i-1] = (un[j-1, i-1] +
                 nu * dt / dx**2 * (un[j-1, i] - 2 * un[j-1, i-1] + un[j-1,i-2]) +
                 nu * dt / dy**2 * (un[j, i-1] - 2 * un[j-1, i-1] + un[j-2,i-1]))

        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

#==============================================================================
# pythran export poisson_2d(float[:,:], float[:,:], float[:,:], int, int, int, float, float)
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

#==============================================================================
# pythran export nonlinearconv_2d(float[:,:], float[:,:], float[:,:], float[:,:], int, float, float, float, float)
def nonlinearconv_2d(u: 'float[:,:]', un: 'float[:,:]',
                     v: 'float[:,:]', vn: 'float[:,:]',
                     nt: int, dt: float, dx: float, dy: float, c: float):

    ###Assign initial conditions
    ##set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
    u[int(.5 / dy):int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2
    ##set hat function I.C. : v(.5<=x<=1 && .5<=y<=1 ) is 2
    v[int(.5 / dy):int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2
    row, col = u.shape

    for n in range(nt + 1): ##loop across number of time steps
        un[:,:] = u[:,:]
        vn[:,:] = v[:,:]
        for j in range(1, row):
            for i in range(1, col):
                u[j, i] = (un[j, i] - (un[j, i] * c * dt / dx * (un[j, i] - un[j, i - 1])) -
                                      (vn[j, i] * c * dt / dy * (un[j, i] - un[j - 1, i])))
                v[j, i] = (vn[j, i] - (un[j, i] * c * dt / dx * (vn[j, i] - vn[j, i - 1])) -
                                      (vn[j, i] * c * dt / dy * (vn[j, i] - vn[j - 1, i])))


        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

        v[0, :] = 1
        v[-1, :] = 1
        v[:, 0] = 1
        v[:, -1] = 1

#==============================================================================
# pythran export laplace_2d(float[:,:], float[:], float, float, float)
def laplace_2d(p: 'float[:,:]', y: 'float[:]',
               dx: float, dy: float, l1norm_target: float):

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

#==============================================================================
# pythran export burgers_2d(float[:,:], float[:,:], float[:,:], float[:,:], int, float, float, float, float)
def burgers_2d(u: 'float[:,:]', un: 'float[:,:]',
               v: 'float[:,:]', vn: 'float[:,:]',
               nt: int, dt: float, dx: float, dy: float, nu: float):

    ###Assign initial conditions
    ##set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
    u[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2
    ##set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
    v[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2
    row, col = u.shape

    for n in range(nt + 1): ##loop across number of time steps
        un[:,:] = u[:,:]
        vn[:,:] = v[:,:]

        for j in range(2, row):
            for i in range(2, col):

                u[j-1, i-1] = (un[i-1, j-1] -
                                 dt / dx * un[j-1, i-1] *
                                 (un[j-1, i-1] - un[j-1, i-2]) -
                                 dt / dy * vn[j-1, i-1] *
                                 (un[j-1, i-1] - un[j-2, i-1]) +
                                 nu * dt / dx**2 *
                                 (un[j-1, i] - 2 * un[j-1, i-1] + un[j-1, i-2]) +
                                 nu * dt / dy**2 *
                                 (un[j, i-1] - 2 * un[j-1, i-1] + un[j-2, i-1]))

                v[j-1, i-1] = (vn[j-1, i-1] -
                                 dt / dx * un[j-1, i-1] *
                                 (vn[j-1, i-1] - vn[j-1, i-2]) -
                                 dt / dy * vn[j-1, i-1] *
                                (vn[j-1, i-1] - vn[j-2, i-1]) +
                                 nu * dt / dx**2 *
                                 (vn[j-1, i] - 2 * vn[j-1, i-1] + vn[j-1, i-2]) +
                                 nu * dt / dy**2 *
                                 (vn[j, i-1] - 2 * vn[j-1, i-1] + vn[j-2, i-1]))

        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

        v[0, :] = 1
        v[-1, :] = 1
        v[:, 0] = 1
        v[:, -1] = 1

#==============================================================================
# pythran export cavity_flow_2d(float[:,:], float[:,:], float[:,:], int, float, float, float, float, float)
def cavity_flow_2d(u: 'float[:,:]', v: 'float[:,:]', p: 'float[:,:]',
                   nt: int, dt: float, dx: float, dy: float,
                   rho: float, nu: float):

    # ...
    def build_up_b(b: 'float[:,:]', rho: float, dt: float, u: 'float[:,:]', v: 'float[:,:]', dx: float, dy: float):
        row, col = p.shape

        for j in range(2, row):
            for i in range(2, col):
                b[j-1, i-1] = (rho * (1 / dt *
                                ((u[j-1, i] - u[j-1, i-2]) /
                                 (2 * dx) + (v[j, i-1] - v[j-2, i-1]) / (2 * dy)) -
                                ((u[j-1, i] - u[j-1, i-2]) / (2 * dx))**2 -
                                  2 * ((u[j, i-1] - u[j-2, i-1]) / (2 * dy) *
                                       (v[j-1, i] - v[j-1, i-2]) / (2 * dx))-
                                      ((v[j, i-1] - v[j-2, i-1]) / (2 * dy))**2))
    # ...

    # ...
    def pressure_poisson(p: 'float[:,:]', dx: float, dy: float, b: 'float[:,:]'):

        row, col = p.shape
        pn = np.empty((row, col))
        # ... copy p to pn
        pn[:,:] = p[:,:]
        # ...

        nit = 50
        for q in range(nit):
            # ... copy p to pn
            pn[:,:] = p[:,:]
            # ...

            for j in range(2, row):
                for i in range(2, col):
                    p[j-1, i-1] = (((pn[j-1, i] + pn[j-1, i-2]) * dy**2 +
                                      (pn[j, i-1] + pn[j-2, i-1]) * dx**2) /
                                      (2 * (dx**2 + dy**2)) -
                                      dx**2 * dy**2 / (2 * (dx**2 + dy**2)) *
                                      b[j-1, i-1])

            p[:, -1] = p[:, -2] # dp/dx = 0 at x = 2
            p[0, :] = p[1, :]   # dp/dy = 0 at y = 0
            p[:, 0] = p[:, 1]   # dp/dx = 0 at x = 0
            p[-1, :] = 0        # p = 0 at y = 2
    # ...

    row, col = p.shape

    un = np.empty((row, col))
    vn = np.empty((row, col))
    b  = np.zeros((row, col))

    for n in range(nt):
        # ... copy u and v to un and vn
        un[:,:] = u[:,:]
        vn[:,:] = v[:,:]
        # ...

        build_up_b(b, rho, dt, u, v, dx, dy)
        pressure_poisson(p, dx, dy, b)

        for j in range(2, row):
            for i in range(2, col):
                u[j-1, i-1] = (un[j-1, i-1]-
                                 un[j-1, i-1] * dt / dx *
                                (un[j-1, i-1] - un[j-1, i-2]) -
                                 vn[j-1, i-1] * dt / dy *
                                (un[j-1, i-1] - un[j-2, i-1]) -
                                 dt / (2 * rho * dx) * (p[j-1, i] - p[j-1, i-2]) +
                                 nu * (dt / dx**2 *
                                (un[j-1, i] - 2 * un[j-1, i-1] + un[j-1, i-2]) +
                                 dt / dy**2 *
                                (un[j, i-1] - 2 * un[j-1, i-1] + un[j-2, i-1])))

                v[j-1, i-1] = (vn[j-1, i-1] -
                                un[j-1, i-1] * dt / dx *
                               (vn[j-1, i-1] - vn[j-1, i-2]) -
                                vn[j-1, i-1] * dt / dy *
                               (vn[j-1, i-1] - vn[j-2, i-1]) -
                                dt / (2 * rho * dy) * (p[j, i-1] - p[j-2, i-1]) +
                                nu * (dt / dx**2 *
                               (vn[j-1, i] - 2 * vn[j-1, i-1] + vn[j-1, i-2]) +
                                dt / dy**2 *
                               (vn[j, i-1] - 2 * vn[j-1, i-1] + vn[j-2, i-1])))

        u[0, :]  = 0
        u[:, 0]  = 0
        u[:, -1] = 0
        u[-1, :] = 1    # set velocity on cavity lid equal to 1
        v[0, :]  = 0
        v[-1, :] = 0
        v[:, 0]  = 0
        v[:, -1] = 0
