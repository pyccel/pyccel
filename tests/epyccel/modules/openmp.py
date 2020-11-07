# pylint: disable=missing-function-docstring, missing-module-docstring/
# pylint: disable=wildcard-import
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

def get_max_threads():
    from pyccel.stdlib.internal.openmp import omp_get_max_threads
    max_threads = omp_get_max_threads()

    return max_threads

@types('int')
def f1(i):
    from pyccel.stdlib.internal.openmp import omp_get_thread_num
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

def test_omp_in_parallel1():
    from pyccel.stdlib.internal.openmp import omp_in_parallel
    in_parallel = omp_in_parallel()
    return in_parallel

def test_omp_in_parallel2():
    from pyccel.stdlib.internal.openmp import omp_in_parallel
    #$ omp parallel
    in_parallel = omp_in_parallel()
    #$ omp end parallel
    return in_parallel

@types ('bool')
def test_omp_set_get_dynamic(dynamic_theads):
    from pyccel.stdlib.internal.openmp import omp_set_dynamic, omp_get_dynamic
    omp_set_dynamic(dynamic_theads)
    return omp_get_dynamic()

@types ('bool')
def test_omp_set_get_nested(nested):
    from pyccel.stdlib.internal.openmp import omp_set_nested, omp_get_nested
    omp_set_nested(nested)
    return omp_get_nested()

def test_omp_get_cancellation():
    from pyccel.stdlib.internal.openmp import omp_get_cancellation
    cancel_var = omp_get_cancellation()
    return cancel_var

def test_omp_get_thread_limit():
    from pyccel.stdlib.internal.openmp import omp_get_thread_limit
    #$ omp parallel
    maximum_threads_available = omp_get_thread_limit()
    #$ omp end parallel
    return maximum_threads_available

@types ('int')
def test_omp_get_set_max_active_levels(max_active_levels):
    from pyccel.stdlib.internal.openmp import omp_get_max_active_levels, omp_set_max_active_levels
    omp_set_max_active_levels(max_active_levels)
    max_active_levels_var = omp_get_max_active_levels()
    return max_active_levels_var

def test_omp_get_level():
    from pyccel.stdlib.internal.openmp import omp_get_level
    #$ omp parallel
    #$ omp parallel
    nested_parallel_regions = omp_get_level()
    #$ omp end parallel
    #$ omp end parallel
    return nested_parallel_regions

def test_omp_get_active_level():
    from pyccel.stdlib.internal.openmp import omp_get_active_level
    #$ omp parallel
    #$ omp parallel
    active_level_vars = omp_get_active_level()
    #$ omp end parallel
    #$ omp end parallel
    return active_level_vars

def test_omp_get_ancestor_thread_num():
    from pyccel.stdlib.internal.openmp import omp_get_ancestor_thread_num, omp_get_active_level
    #$ omp parallel
    active_level = omp_get_active_level()
    ancestor_thread = omp_get_ancestor_thread_num(active_level)
    #$ omp end parallel
    return ancestor_thread

def test_omp_get_team_size():
    from pyccel.stdlib.internal.openmp import omp_get_team_size, omp_get_active_level
    #$ omp parallel
    active_level = omp_get_active_level()
    team_size = omp_get_team_size(active_level)
    #$ omp end parallel
    return team_size

def test_omp_in_final():
    from pyccel.stdlib.internal.openmp import omp_in_final
    x = 20
    z = 0
    result = 0

    #$ omp parallel
    #$ omp single
    #$ omp task final (i >= 10)
    for i in range(x):
        z = z + i
        if omp_in_final() == 1:
            result = 1
    #$ omp end task
    #$ omp end single
    #$ omp end parallel
    return result

def test_omp_get_proc_bind():
    from pyccel.stdlib.internal.openmp import omp_get_proc_bind

    bind_var = omp_get_proc_bind()
    return bind_var

#The function give som errors
# def test_omp_places():
#     from pyccel.stdlib.internal.openmp import omp_get_partition_num_places
#     from pyccel.stdlib.internal.openmp import omp_get_partition_place_nums
#     from pyccel.stdlib.internal.openmp import omp_get_place_num
#     from pyccel.stdlib.internal.openmp import omp_get_place_proc_ids
#     from pyccel.stdlib.internal.openmp import omp_get_place_num_procs
#     from pyccel.stdlib.internal.openmp import omp_get_num_places
#
#     partition_num_places = omp_get_partition_num_places()
#     #partition_places_num =    omp_get_partition_place_nums(0)
#     place_num = omp_get_place_num()
#     if place_num < 0:
#         return -1
#     #place_num, ids = omp_get_place_proc_ids(place_num, ids)
#     procs = omp_get_place_num_procs(place_num)
#     num_places = omp_get_num_places()
#     return place_num

@types ('int')
def test_omp_set_get_default_device(device_num):
    from pyccel.stdlib.internal.openmp import omp_get_default_device
    from pyccel.stdlib.internal.openmp import omp_set_default_device
    omp_set_default_device(device_num)
    default_device = omp_get_default_device()
    return default_device

def test_omp_get_num_devices():
    from pyccel.stdlib.internal.openmp import omp_get_num_devices
    num_devices = omp_get_num_devices()
    return num_devices

def test_omp_get_num_teams():
    from pyccel.stdlib.internal.openmp import omp_get_num_teams
    #$ omp teams num_teams(2)
    num_teams = omp_get_num_teams()
    #$ omp end teams
    return num_teams

@types('int')
def test_omp_get_team_num(i):
    from pyccel.stdlib.internal.openmp import omp_get_team_num
    out = -1
    #$ omp teams num_teams(2)
    team_num = omp_get_team_num()
    if team_num == i:
        out = team_num
    #$ omp end teams
    return out

def test_omp_is_initial_device():
    from pyccel.stdlib.internal.openmp import omp_is_initial_device
    is_task_in_init_device = omp_is_initial_device()
    return is_task_in_init_device

def test_omp_get_initial_device():
    from pyccel.stdlib.internal.openmp import omp_get_initial_device
    #$ omp target device(deviceid)
    host_device = omp_get_initial_device()
    #$ omp end target
    return host_device

def test_omp_get_set_schedule():
    from pyccel.stdlib.internal.openmp import omp_get_schedule, omp_set_schedule
    result = 0
    #$ omp parallel private(i)
    #$ omp do schedule(runtime) reduction (+:sum)
    omp_set_schedule(2, 2)
    schedule_kind = 0
    chunk_size = 0
    omp_get_schedule(schedule_kind, chunk_size)
    for i in range(16):
        result = result + i
    #$ omp end do nowait
    return True

def test_omp_get_max_task_priority():
    from pyccel.stdlib.internal.openmp import omp_get_max_task_priority
    result = 0
    #$ omp parallel
    #$ omp single
    #$ omp task priority(i)
    for i in range(10):
        result = result + i
        if i == 5:
            max_task_priority_var = omp_get_max_task_priority()
    #$ omp end single
    #$ omp end parallel
    return max_task_priority_var

@types('real[:,:], real[:,:], real[:,:]')
def omp_matmul(A, x, out):
    #$ omp parallel shared(A,x,out) private(i,j,k)
    #$ omp do
    for i in range(len(A)):# pylint: disable=C0200
        for j in range(len(x[0])):# pylint: disable=C0200
            for k in range(len(x)):# pylint: disable=C0200
                out[i][j] += A[i][k] * x[k][j]
    #$ omp end do
    #$ omp end parallel
    #to let the function compile using epyccel issue #468
    "bypass issue #468" # pylint: disable=W0105

@types('real[:,:], real[:,:], real[:,:]')
def omp_matmul_single(A, x, out):
    from numpy import matmul
    #$ omp parallel
    #$ omp single
    out[:] = matmul(A, x)
    #$ omp end single
    #$ omp end parallel
    #to let the function compile using epyccel issue #468
    "bypass issue #468" # pylint: disable=W0105

@types('int[:]')
def omp_arraysum(x):
    result = 0
    #$ omp parallel private(i)
    #$ omp do reduction (+:result)
    for i in range(0, 5):
        result += x[i]
    #$ omp end do
    #$ omp end parallel
    return result

@types('int[:]')
def omp_arraysum_single(x):
    result = 0
    #$ omp parallel
    #$ omp single
    for i in range(0, 10):
        result += x[i]
    #$ omp end single
    #$ omp end parallel
    return result
