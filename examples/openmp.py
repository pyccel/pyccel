from PIL import Image
from pyccel.epyccel import epyccel
from numpy import random
from pyccel.decorators import types
import numpy as np
import time

arr = np.zeros([1000, 1000], dtype=int)
arr1 = np.copy(arr)
arr2 = np.copy(arr)
arr3 = np.copy(arr)
@types('int[:,:]','int')
def openmp_ex1(arr, it):
    p = 0
    #$ omp parallel num_threads(2) private(p, x, y, c_i, c_r, z_i, z_r, i)
    #$ omp for
    for p in range(0, (1000 * 1000)):
        x = int(p % 1000)
        y = int(p / 1000)
        c_i = (y - 500) / 200
        c_r = (x - 500) / 200
        z_i = 0.0
        z_r = 0.0
        i = 0
        while (z_r * z_r + z_i * z_i < 4 and i < it):
            tmp = z_r
            z_r = z_r * z_r - z_i * z_i + c_r
            z_i = 2 * tmp * z_i + c_i
            i += 1
        if i != it:
            arr[y,x] = 1
    #$ omp end parallel
    "by pass issue"

omp_f1 = epyccel(openmp_ex1, accelerator='openmp', language='fortran', verbose=True)
#f1 = epyccel(openmp_ex1, language='c')

start = time.time()
omp_f1(arr1, 100000)
end = time.time()
omp_time = end - start

#start = time.time()
#f1(arr2, 100000)
#end = time.time()
#c_time = end - start

print("Time using OpenMP: ", omp_time)
#print("Time without OpenMP: ", c_time)
#openmp_ex1(arr3, 100)

img1 = Image.frombytes('1', arr1.shape[::-1], np.packbits(arr1, 1))
img1.show(title="C_OpenMP")

#img2 = Image.frombytes('1', arr2.shape[::-1], np.packbits(arr2, 1))
#img2.show(title="C")

