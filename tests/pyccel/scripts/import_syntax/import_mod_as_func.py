# pylint: disable=missing-function-docstring, missing-module-docstring

#$ header function matmat(float [:, :], float [:, :], float [:, :])
def matmat(a, b, c):
    import numpy as np #pylint: disable=W0404
    n, m = np.shape(a)
    m, p = np.shape(b)

    for i in range(0, n):
        for j in range(0, p):
            for k in range(0, m):
                c[i, j] = c[i, j] + a[i, k]*b[k, j]

if __name__ == '__main__':
    import numpy as np #pylint: disable=W0404

    n = 3
    m = 4
    p = 3

    a = np.zeros((n, m), 'float')
    b = np.zeros((m, p), 'float')

    for i in range(0, n):
        for j in range(0, m):
            a[i, j] = (i-j)*1.0
            print(a[i, j])
        print()
    print()

    for i in range(0, m):
        for j in range(0, p):
            b[i, j] = (i+j)*1.0
            print(b[i, j])
        print()
    print()

    c = np.zeros((n, p), 'float')
    matmat(a, b, c)

    for i in range(0, n):
        for j in range(0, p):
            print(c[i, j])
        print()
