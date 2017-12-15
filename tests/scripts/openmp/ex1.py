# coding: utf-8

#  export OMP_NUM_THREADS=2

from pyccelext.math.external.openmp import omp_get_num_threads
from pyccelext.math.external.openmp import omp_get_max_threads
from pyccelext.math.external.openmp import omp_get_thread_num


#$ omp parallel

n_threads = omp_get_num_threads()
print("> threads number : ", n_threads)

max_threads = omp_get_max_threads()
print("> maximum available threads : ", max_threads)

idx = omp_get_thread_num()
print("> thread  id : ", idx)

#$ omp end parallel
