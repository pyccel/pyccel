# coding: utf-8

from pyccel.stdlib.parallel.openacc import Parallel

with Parallel(numGangs=2, private=['idx']):
    idx = 0
    print("> hello")
