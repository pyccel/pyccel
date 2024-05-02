# coding: utf-8

from pyccel.stdlib.parallel.openmp import Range
from pyccel.stdlib.parallel.openmp import Parallel

x = 0.0

with Parallel(num_threads=2, default='shared'):
    for i in Range(-2, 5, 1, nowait=True, private=['i'], reduction=['+', 'x'], schedule='static', ordered=True):
        x += 2 * i

print('x = ', x)
