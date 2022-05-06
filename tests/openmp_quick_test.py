
def test_omp_get_max_task_priority():
    import numpy as np
    from pyccel.stdlib.internal.openmp import omp_get_max_task_priority
    max_task_priority_var = np.int32(0)
    #$ omp parallel
    #$ omp single
    #$ omp task
    max_task_priority_var = omp_get_max_task_priority()
    #$ omp end task
    #$ omp end single
    #$ omp end parallel
    return max_task_priority_var
