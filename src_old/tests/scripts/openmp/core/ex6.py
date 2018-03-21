# coding: utf-8

# This example is the python implementation of collapse.3.f from OpenMP 4.5 examples

from pyccel.stdlib.parallel.openmp import omp_get_thread_num

#$ omp parallel num_threads(2)

#$ omp do collapse(2) ordered private(j,k) schedule(static,3)
for k in range(0,3):
    for j in range(0,2):

#$ omp ordered
        t_id = omp_get_thread_num()
        print(t_id, k, j)
#$ omp end ordered

        # call work(a,j,k)
#$ omp end do

#$ omp end parallel
