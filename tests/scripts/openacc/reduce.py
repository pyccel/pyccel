# coding: utf-8

from pyccel.stdlib.parallel.openacc import Range
from pyccel.stdlib.parallel.openacc import Parallel

x = 0.0

with Parallel(num_gangs=2):
    for i in Range(-2, 5, 1, private=['i'], reduction=['+', 'x']):
        x += 2 * i

print('x = ', x)
