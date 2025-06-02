# pylint: disable=missing-function-docstring, missing-module-docstring
def parallel_if(n : int):
    import numpy as np
    from pyccel.stdlib.internal.openmp import omp_get_thread_num, omp_get_num_threads
    a = np.zeros(n)
    thid, nthrds =  np.int32(0), np.int32(0)
    start, end = 0, 0

    #$ omp parallel if(parallel:n > 10) private(thid, nthrds, start, end) num_threads(4)
    thid = omp_get_thread_num()
    nthrds = omp_get_num_threads()

    start = int(thid * n / nthrds)
    end = int((thid + 1) * n / nthrds)
    for i in range(start, end):
        a[i] = 2 * i
    #$ omp end parallel
    return a
