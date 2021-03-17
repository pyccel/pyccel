from pyccel.epyccel import epyccel
from numpy import random
import numpy as np
arr = np.zeros((250000,), dtype=int)
def openmp_ex1():
    x1 = -2.1
    p = 0
    for p in range(0, (500 * 500)):
        x = p % 500
        y = p / 500
        c_i = (y - 250) / 100
        c_r = (x - 250) / 100
        z_i = 0
        z_r = 0
        i = 0
    return p

f1 = epyccel(openmp_ex1, accelerator='openmp', language='c')

print(f1())
