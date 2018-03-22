# coding: utf-8

from pyccel.stdlib.parallel.openacc import Parallel

with Parallel(num_gangs=2, private=['idx']):
    idx = 0
    print("> hello")
