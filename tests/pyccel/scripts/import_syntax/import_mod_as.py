# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np

#$ header function matmat(double [:,:], double [:,:], double [:,:])
def matmat(a,b,c):
    n, m = np.shape(a)
    m, p = np.shape(b)

    for i in range(0, n):
        for j in range(0, p):
            for k in range(0, m):
                c[i,j] = c[i,j] + a[i,k]*b[k,j]

if __name__ == '__main__':
    n = 3
    m = 4
    p = 3

    a = np.zeros((n,m), 'double')
    b = np.zeros((m,p), 'double')

    for i in range(0, n):
        for j in range(0, m):
            a[i,j] = (i-j)*1.0
            print(a[i,j])
        print()
    print()

    for i in range(0, m):
        for j in range(0, p):
            b[i,j] = (i+j)*1.0
            print(b[i,j])
        print()
    print()

    c = np.zeros((n,p),'double')
    matmat(a,b,c)

    for i in range(0, n):
        for j in range(0, p):
            print(c[i,j])
        print()
