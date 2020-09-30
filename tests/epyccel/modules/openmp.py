from pyccel.decorators import types

@types(int)
def set_num_threads(n):
    from pyccel.stdlib.internal.openmp import omp_set_num_threads
    omp_set_num_threads(n)

@types('int')
def f1(i):
    from pyccel.stdlib.internal.openmp import omp_get_num_threads
    from pyccel.stdlib.internal.openmp import omp_get_max_threads
    from pyccel.stdlib.internal.openmp import omp_get_thread_num

    n_threads   = omp_get_num_threads()
    max_threads = omp_get_max_threads()

    out = -1
    #$ omp parallel private(idx)

    idx = omp_get_thread_num()
    
    if idx == i:
        out = idx

    #$ omp end parallel
    return out
  
def test_all_omp_funcs()
{
    from pyccel.stdlib.internal.openmp import *

    procs_num = omp_get_num_procs()
    print("the number of processors available to this device is:".procs_number)
}