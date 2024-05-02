
#$ header function matmat(double [:, :], double [:, :], double [:, :])
def matmat(a, b, c):
    import numpy as np #pylint: disable=W0404
    nm = np.shape(a)
    mp = np.shape(b)

    n = nm[0]
    m = nm[1]
    p = mp[1]

    for i in range(0, n):
        for j in range(0, p):
            for k in range(0, m):
                c[i, j] = c[i, j] + a[i, k]*b[k, j]

import numpy as np #pylint: disable=W0404

n = 3
m = 4
p = 3

a = np.zeros((n, m), 'double')
b = np.zeros((m, p), 'double')

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

c = np.zeros((n, p), 'double')
matmat(a, b, c)

for i in range(0, n):
    for j in range(0, p):
        print(c[i, j])
    print()
