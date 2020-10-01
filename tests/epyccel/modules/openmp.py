from pyccel.decorators import types

@types(int)
def set_num_threads(n):
    from pyccel.stdlib.internal.openmp import omp_set_num_threads
    omp_set_num_threads(n)

@types()
def get_num_threads():
    from pyccel.stdlib.internal.openmp import omp_get_num_threads
    #$ omp parallel
    n = omp_get_num_threads()
    #$ omp end parallel
    return n

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

def test_omp_number_of_procs():
    from pyccel.stdlib.internal.openmp import omp_get_num_procs

    procs_num = omp_get_num_procs()
    return procs_num

def test_omp_in_parallel():
    from pyccel.stdlib.internal.openmp import omp_in_parallel

    in_parallel = omp_in_parallel()
    return in_parallel

@types ('bool')
def test_omp_set_get_dynamic(dynamic_theads):
    from pyccel.stdlib.internal.openmp import omp_set_dynamic
    from pyccel.stdlib.internal.openmp import omp_get_dynamic

    omp_set_dynamic(dynamic_theads)
    return omp_get_dynamic()
