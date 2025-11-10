
def set_num_threads(n : int):
    import numpy as np
    from pyccel.stdlib.internal.openmp import omp_set_num_threads
    omp_set_num_threads(np.int32(n))
