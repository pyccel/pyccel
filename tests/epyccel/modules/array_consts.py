# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np

a = np.ones(6)
b = np.array([1,2,3,4,5])
c = np.zeros((2,3), dtype=np.int32)
d = np.array([1+2j, 3+4j])
e = np.empty((2,3,4))

for i in range(2):
    for j in range(3):
        for k in range(4):
            e[i,j,k] = i*12+j*4+k

def update_a():
    a[:] = a+1

def reset_a():
    a[:] = 1
