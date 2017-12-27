# coding: utf-8

from pyccel.stdlib.parallel.openmp import omp_get_thread_num
from pyccel.stdlib.parallel.openmp import Parallel

with Parallel(num_threads=2, default='shared', private=['idx']):
    idx = omp_get_thread_num()

    print("> thread  id : ", idx)
