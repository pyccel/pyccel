# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

from pyccel.stdlib.internal.openmp import omp_get_num_threads
from pyccel.stdlib.internal.openmp import omp_get_max_threads
from pyccel.stdlib.internal.openmp import omp_get_thread_num

n_threads = omp_get_num_threads()
print("> threads number : ", n_threads)

max_threads = omp_get_max_threads()
print("> maximum available threads : ", max_threads)

#$ omp parallel private(idx)

idx = omp_get_thread_num()
print("> thread  id : ", idx)

#$ omp end parallel
