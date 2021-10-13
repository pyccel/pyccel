from numba import njit
import numpy as np

#==============================================================================
@njit(fastmath=True)
def linearconv_1d(u: 'float[:]', un: 'float[:]',
                  nt: int, nx: int,
                  dt: float, dx: float, c: float):

    for n in range(nt):
        un[:nx] = u[:nx]

        for i in range(1, nx):
            u[i] = un[i] - c * dt / dx * (un[i] - un[i-1])

#==============================================================================
@njit(fastmath=True)
def lineardiff_1d(u: 'float[:]', un: 'float[:]',
                  nt: int, nx: int,
                  dt: float, dx: float, nu: float):

    for n in range(nt):
        un[:] = u[:]
        for i in range(1, nx - 1):
            u[i] = un[i] + nu * dt / dx**2 * (un[i+1] - 2 * un[i] + un[i-1])

#==============================================================================
@njit(fastmath=True)
def nonlinearconv_1d(u: 'float[:]', un: 'float[:]',
                     nt: int, nx: int, dt: float, dx: float):

    for n in range(nt):
        un[:] = u[:]
        for i in range(1, nx):
            u[i] = un[i] - un[i] * dt / dx * (un[i] - un[i-1])

#==============================================================================
@njit(fastmath=True)
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
@njit(fastmath=True)
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
@njit(fastmath=True)
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

#==============================================================================
@njit(fastmath=True)
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
@njit(fastmath=True)
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
@njit(fastmath=True)
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
@njit(fastmath=True)
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

@njit(fastmath=True)
def test_linearconv_1d(nx : int =2001, nt : int =2000, c : float =1., dt :float =0.0003):

    # ...
    dx = 2 / (nx-1)
    grid = np.linspace(0,2,nx)

    u0 = np.ones(nx)
    u0[int(.5 / dx):int(1 / dx + 1)] = 2
    u = np.empty(nx)
    u[:] = u0
    un = np.ones(nx)
    # ...

    # ...
    linearconv_1d(u, un, nt, nx, dt, dx, c)
    # ...

#==============================================================================
@njit(fastmath=True)
def test_lineardiff_1d(nx : int = 501, nt : int = 1500, nu : float = 0.3):

    # ...
    dx = 2 / (nx - 1)
    CFL = .5
    dt = CFL * dx**2 / nu
    grid = np.linspace(0,2,nx)

    u0 = np.ones(nx)      #a numpy array with nx elements all equal to 1.
    u0[int(.5 / dx):int(1 / dx + 1)] = 2  #setting u = 2 between 0.5 and 1 as per our I.C.s
    u = np.empty(nx)
    u[:] = u0
    un = np.ones(nx) #our placeholder array, un, to advance the solution in time
    # ...

    # ...
    lineardiff_1d(u, un, nt, nx, dt, dx, nu)
    # ...

#==============================================================================
@njit(fastmath=True)
def test_nonlinearconv_1d(nx : int =2001, nt : int =2000, c : float =1., dt : float =0.00035):

    # ...
    dx = 2 / (nx-1)
    grid = np.linspace(0,2,nx)

    u0 = np.ones(nx)
    u0[int(.5 / dx):int(1 / dx + 1)] = 2
    u = np.empty(nx)
    u[:] = u0
    un = np.ones(nx)
    # ...

    # ...
    nonlinearconv_1d(u, un, nt, nx, dt, dx)
    # ...

"""
#==============================================================================
@njit(fastmath=True)
def test_burgers_1d(nx : int =201, nt : int =100, nu : float=0.07):

    def ufunc(t,x,nu):
        (4*(t + 1)*(exp((4*t - x)**2/(4*nu*(t + 1))) + exp((4*t - x + 2*pi)**2/(4*nu*(t + 1)))) - (4*t - x)*exp((4*t - x + 2*pi)**2/(4*nu*(t + 1))) - (4*t - x + 2*pi)*exp((4*t - x)**2/(4*nu*(t + 1))))/((t + 1)*(exp((4*t - x)**2/(4*nu*(t + 1))) + exp((4*t - x + 2*pi)**2/(4*nu*(t + 1)))))

    # ...
    def analytical_solution():
        x, nu, t = sp.symbols('x nu t')
        phi = (sp.exp(-(x - 4 * t)**2 / (4 * nu * (t + 1))) +
               sp.exp(-(x - 4 * t - 2 * sp.pi)**2 / (4 * nu * (t + 1))))

        phiprime = phi.diff(x)
        u = -2 * nu * (phiprime / phi) + 4
        ufunc = lambdify((t, x, nu), u)
        return ufunc

    dx = 2 * np.pi / (nx - 1)
    dt = dx * nu

    x = np.linspace(0, 2 * np.pi, nx)
    un = np.empty(nx)
    t = 0

    ufunc = analytical_solution()
    u = np.asarray([ufunc(t, x0, nu) for x0 in x])
    # ...

    # ...
    burgers_1d(u, un, nt, nx, dt, dx, nu)
    # ...
"""
#==============================================================================
@njit(fastmath=True)
def test_linearconv_2d(nx : int = 201, ny : int = 201, nt : int = 100, c : float = 1.):

    # ...
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    sigma = .2
    dt = sigma * dx

    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)

    u0 = np.ones((ny, nx))
    u0[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2
    u = np.empty((ny,nx))
    u[:] = u0
    un = np.ones((ny, nx)) ##
    # ...

    # ...
    linearconv_2d(u, un, nt, dt, dx, dy, c)
    # ...

#==============================================================================
@njit(fastmath=True)
def test_lineardiff_2d(nt : int = 51, nx : int = 101, ny : int = 101, nu : float = .05):

    # ...
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    sigma = .25
    dt = sigma * dx * dy / nu

    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)

    u = np.ones((ny, nx))  # create a 1xn vector of 1's
    un = np.ones((ny, nx))
    # ...

    # ...
    lineardiff_2d(u, un, nt, dt, dx, dy, nu)
    # ...

#==============================================================================
@njit(fastmath=True)
def test_poisson_2d(nx : int = 150, ny : int = 150, nt : int = 100):

    # ...
    xmin = 0
    xmax = 2
    ymin = 0
    ymax = 1

    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)

    # Initialization
    p  = np.zeros((ny, nx))
    pd = np.zeros((ny, nx))
    b  = np.zeros((ny, nx))
    x  = np.linspace(xmin, xmax, nx)
    y  = np.linspace(xmin, xmax, ny)
    # ...

    # ...
    poisson_2d(p, pd, b, nx, ny, nt, dx, dy)
    # ...

#==============================================================================
@njit(fastmath=True)
def test_laplace_2d(nx : int = 31, ny : int = 31, c : float = 1., l1norm_target : float =1.e-4):

    # ...
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)

    p = np.zeros((ny, nx))  # create a XxY vector of 0's

    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 1, ny)

    p[:, 0] = 0  # p = 0 @ x = 0
    p[:, -1] = y  # p = y @ x = 2
    p[0, :] = p[1, :]  # dp/dy = 0 @ y = 0
    p[-1, :] = p[-2, :]  # dp/dy = 0 @ y = 1
    # ...

    # ...
    laplace_2d(p, y, dx, dy, l1norm_target)
    # ...

#==============================================================================
@njit(fastmath=True)
def test_nonlinearconv_2d(nx : int = 201, ny : int = 201, nt : int = 80, c : float = 1.):

    # ...
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    sigma = .2
    dt = sigma * dx

    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)

    u = np.ones((ny, nx)) ##create a 1xn vector of 1's
    v = np.ones((ny, nx))
    un = np.ones((ny, nx))
    vn = np.ones((ny, nx))
    # ...

    # ...
    nonlinearconv_2d(u, un, v, vn, nt, dt, dx, dy, c)
    # ...

#==============================================================================
@njit(fastmath=True)
def test_burgers_2d(nx : int = 41, ny : int = 41, nt : int = 420, nu : float = 0.01):

    # ...
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    sigma = .0009
    dt = sigma * dx * dy / nu

    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)

    u = np.ones((ny, nx))  # create a 1xn vector of 1's
    v = np.ones((ny, nx))
    un = np.ones((ny, nx))
    vn = np.ones((ny, nx))
    # ...

    # ...
    burgers_2d(u, un, v, vn, nt, dt, dx, dy, nu)
    # ...

#==============================================================================
@njit(fastmath=True)
def test_cavity_flow_2d(nx : int = 41, ny : int = 41, nt : int = 500, rho : float = 1., nu : float = .1, dt : float = .001):

    # ...
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)

    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    b = np.zeros((ny, nx))
    # ...

    # ...
    cavity_flow_2d(u, v, p, nt, dt, dx, dy, rho, nu)
    # ...
