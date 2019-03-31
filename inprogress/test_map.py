from pyccel.decorators import types

@types('int[:]','int[:]','int','int[:,:]')
def map_f(arr_x, arr_y, z, arr_r):


    @types('int','int','int')
    def f(x, y, z):


        r = x + y*z
        return r

    len_arr_x = len(arr_x)
    len_arr_y = len(arr_y)
    #$ omp parallel
    #$ omp do schedule(runtime)
    for i_y in range(0, len_arr_y, 1):
        ly = arr_y[i_y]
        for i_x in range(0, len_arr_x, 1):
            lx = arr_x[i_x]
            arr_r[i_x, i_y] = f(lx,ly,z)

    #$ omp end do nowait
    #$ omp end parallel

    print('DONE')


from numpy import zeros
nx = 5
ny = 4
xs = zeros(nx, 'int')
ys = zeros(ny, 'int')
rs = zeros((nx,ny), 'int')

xs = [1, 2, 3, 4, 5]
ys = [1, 2, 3, 4]

map_f(xs, ys, 2, rs)
