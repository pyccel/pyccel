# pylint: disable=missing-function-docstring, missing-module-docstring

def matmul(a: 'float[:,:](order=C)',
           b: 'float[:,:](order=F)',
           c: 'float[:,:](order=C)'):

    m, p = a.shape
    q, n = b.shape
    r, s = c.shape

    if p != q or m != r or n != s:
        return -1

    for i in range(m):
        for j in range(n):
            c[i, j] = 0.0
            for k in range(p):
                c[i, j] += a[i, k] * b[k, j]

    return 0

