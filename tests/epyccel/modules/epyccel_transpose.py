# pylint: disable=missing-function-docstring, missing-module-docstring

def transpose_shape_1(x: "int[:,:]"):
    from numpy import transpose

    y = transpose(x)
    n, m = y.shape
    return n, m, y[-1, 0], y[0, -1]

def transpose_shape_2(x: "int[:,:,:]"):
    from numpy import transpose

    y = transpose(x)
    n, m, p = y.shape
    return n, m, p, y[0, -1, 0], y[0, 0, -1], y[-1, -1, 0]

def transpose_property_1(x: "int[:,:]"):
    y = x.T
    n, m = y.shape
    return n, m, y[-1, 0], y[0, -1]

def transpose_property_2(x: "int[:,:,:]"):
    y = x.T
    n, m, p = y.shape
    return n, m, p, y[0, -1, 0], y[0, 0, -1], y[-1, -1, 0]

def transpose_in_expression_1(x: "int[:,:]"):
    from numpy import transpose

    y = transpose(x) + 3
    n, m = y.shape
    return n, m, y[-1, 0], y[0, -1]

def transpose_in_expression_2(x: "int[:,:,:]"):
    y = x.T * 3
    n, m, p = y.shape
    return n, m, p, y[0, -1, 0], y[0, 0, -1], y[-1, -1, 0]

def mixed_order_1(x: "int[:,:]"):
    from numpy import ones, transpose

    n, m = x.shape
    y = ones((m, n), order="F")
    z = x + transpose(y)
    n, m = z.shape
    return n, m, z[-1, 0], z[0, -1]

def mixed_order_2(x: "int[:,:]"):
    from numpy import ones

    n, m = x.shape
    y = ones((m, n), order="F")
    z = x.transpose() + y
    n, m = z.shape
    return n, m, z[-1, 0], z[0, -1]

def mixed_order_3(x: "int[:,:,:]"):
    from numpy import ones, transpose

    n, m, p = x.shape
    y = ones((p, m, n))
    z = transpose(x) + y
    n, m, p = z.shape
    return n, m, p, z[0, -1, 0], z[0, 0, -1], z[-1, -1, 0]

def transpose_pointer_1(x: "int[:,:]"):
    from numpy import transpose

    y = transpose(x)
    x[0, -1] += 22
    n, m = y.shape
    return n, m, y[-1, 0], y[0, -1]

def transpose_pointer_2(x: "int[:,:,:]"):
    y = x.T
    x[0, -1, 0] += 11
    n, m, p = y.shape
    return n, m, p, y[0, -1, 0], y[0, 0, -1], y[-1, -1, 0]

def transpose_of_expression_1(x: "int[:,:]"):
    from numpy import transpose

    y = transpose(x * 2) + 3
    n, m = y.shape
    return n, m, y[-1, 0], y[0, -1]

def transpose_of_expression_2(x: "int[:,:,:]"):
    y = (x * 2).T * 3
    n, m, p = y.shape
    return n, m, p, y[0, -1, 0], y[0, 0, -1], y[-1, -1, 0]

def force_transpose_1(x: "int[:,:]"):
    from numpy import transpose, empty

    n, m = x.shape
    y = empty((m, n))
    y[:, :] = transpose(x)
    n, m = y.shape
    return n, m, y[-1, 0], y[0, -1]

def force_transpose_2(x: "int[:,:,:]"):
    from numpy import empty

    n, m, p = x.shape
    y = empty((p, m, n))
    y[:, :, :] = x.transpose()
    n, m, p = y.shape
    return n, m, p, y[0, -1, 0], y[0, 0, -1], y[-1, -1, 0]

def transpose_to_inner_indexes_1(x: "int[:,:]", y: "int[:,:,:,:]"):
    y[0, :, :, 0] = x.T

def transpose_to_inner_indexes_2(x: "int[:,:]", y: "int[:,:,:,:,:]"):
    y[0, :, 0, :, 0] = x.T

def transpose_to_inner_indexes_3(x: "int[:,:,:]", y: "int[:,:,:,:,:]"):
    y[0, :, :, :, 0] = x.T
