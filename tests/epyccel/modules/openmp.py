# pylint: disable=missing-function-docstring, missing-module-docstring

def set_num_threads(n : int):
    import numpy as np
    from pyccel.stdlib.internal.openmp import omp_set_num_threads
    omp_set_num_threads(np.int32(n))

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

def f1(i : 'int'):
    from pyccel.stdlib.internal.openmp import omp_get_thread_num
    out = -1
    #$ omp parallel private(idx)
    idx = omp_get_thread_num()

    if idx == i:
        out = int(idx)

    #$ omp end parallel
    return out

def directive_in_else(x : int):
    func_result = 0
    if x < 30:
        return x
    else:
        #$ omp parallel
        #$ omp for reduction(+:func_result)
        for i in range(x):
            func_result = func_result + i
        #$ omp end parallel

    return func_result

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

def test_omp_set_get_dynamic(dynamic_theads : 'bool'):
    from pyccel.stdlib.internal.openmp import omp_set_dynamic, omp_get_dynamic
    omp_set_dynamic(dynamic_theads)
    return omp_get_dynamic()

def test_omp_set_get_nested(nested : 'bool'):
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

def test_omp_get_set_max_active_levels(max_active_levels : 'int'):
    import numpy as np
    from pyccel.stdlib.internal.openmp import omp_get_max_active_levels, omp_set_max_active_levels
    omp_set_max_active_levels(np.int32(max_active_levels))
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
    func_result = 0

    #$ omp parallel
    #$ omp single
    #$ omp task final(i >= 10)
    for i in range(x):
        z = z + i
        if omp_in_final() == 1:
            func_result = 1
    #$ omp end task
    #$ omp end single
    #$ omp end parallel
    return func_result

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

def test_omp_set_get_default_device(device_num : 'int'):
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

def test_omp_get_team_num(i : 'int'):
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
    #$ omp target
    host_device = omp_get_initial_device()
    #$ omp end target
    return host_device

def test_omp_get_set_schedule():
    import numpy as np
    from pyccel.stdlib.internal.openmp import omp_get_schedule, omp_set_schedule
    func_result = 0
    #$ omp parallel private(i)
    omp_set_schedule(np.int32(2), np.int32(3))
    _, chunk_size = omp_get_schedule()
    #$ omp for nowait schedule(runtime) reduction (+:func_result)
    for i in range(16):
        func_result = func_result + chunk_size
    #$omp end parallel
    return func_result

def test_nowait_schedule(n : int):
    import numpy as np
    from pyccel.stdlib.internal.openmp import omp_get_thread_num, omp_get_num_threads

    a = np.zeros(n)
    imin_res = np.empty(4)
    imax_res = np.empty(4)

    #$omp parallel private(rank,nb_tasks,i_min,i_max)
    rank = omp_get_thread_num()
    nb_tasks=omp_get_num_threads()
    i_min=n
    i_max=0

    schedule_size = int(n/nb_tasks) #pylint: disable=unused-variable
    #$omp for nowait schedule(static, schedule_size)
    for i in range(n):
        a[i] = 92290. + i
        i_min = min(i_min, i)
        i_max = max(i_min, i)

    imin_res[rank] = i_min
    imax_res[rank] = i_max
    #$omp end parallel

    return imin_res[0], imin_res[1], imin_res[2], imin_res[3], \
            imax_res[0], imax_res[1], imax_res[2], imax_res[3]

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

def omp_matmul(A : 'float[:,:]', x : 'float[:,:]', out : 'float[:,:]'):
    #$ omp parallel shared(A,x,out) private(i,j,k)
    #$ omp for
    for i in range(len(A)):# pylint: disable=C0200
        for j in range(len(x[0])):# pylint: disable=C0200
            for k in range(len(x)):# pylint: disable=C0200
                out[i][j] += A[i][k] * x[k][j]
    #$ omp end parallel
    #to let the function compile using epyccel issue #468
    "bypass issue #468" # pylint: disable=W0105

def omp_matmul_single(A : 'float[:,:]', x : 'float[:,:]', out : 'float[:,:]'):
    from numpy import matmul
    #$ omp parallel
    #$ omp single
    out[:] = matmul(A, x)
    #$ omp end single
    #$ omp end parallel
    #to let the function compile using epyccel issue #468
    "bypass issue #468" # pylint: disable=W0105


def omp_nowait(x : 'int[:]', y : 'int[:]', z : 'float[:]'):
    #$ omp parallel
    #$ omp for nowait
    for i in range(0, 1000):
        y[i] = x[i] * 2
    #$ omp for nowait
    for j in range(0, 1000):
        z[j] = x[j] / 2
    #$ omp end parallel
    "bypass issue #468" # pylint: disable=W0105

def omp_arraysum(x : 'int[:]'):
    func_result = 0
    #$ omp parallel private(i)
    #$ omp for reduction (+:func_result)
    for i in range(0, 5):
        func_result += x[i]
    #$ omp end parallel
    return func_result

def omp_arraysum_combined(x : 'int[:]'):
    func_result = 0
    #$ omp parallel for reduction (+:func_result)
    for i in range(0, 5):
        func_result += x[i]
    return func_result

def omp_range_sum_critical(x : 'int'):
    func_result = 0
    #$ omp parallel for num_threads(4) shared(func_result)
    for i in range(0, x):
        #$ omp critical
        func_result += i
        #$ omp end critical
    return func_result


def omp_arraysum_single(x : 'int[:]'):
    func_result = 0
    #$ omp parallel
    #$ omp single
    for i in range(0, 10):
        func_result += x[i]
    #$ omp end single
    #$ omp end parallel
    return func_result

def omp_master():
    func_result = 30
    #$omp parallel num_threads(3) reduction(+:func_result)
    #$omp master
    func_result += 1
    #$omp end master
    #$omp end parallel
    return func_result

def omp_taskloop(n : 'int'):
    func_result = 0
    #$omp parallel num_threads(n)
    #$omp taskloop
    for i in range(0, 10): # pylint: disable=unused-variable
        #$omp atomic
        func_result = func_result + 1
    #$omp end parallel
    return func_result

def omp_tasks(x : 'int'):
    def fib(n : int) -> int:
        if n < 2:
            return n
        #$ omp task shared(i) firstprivate(n)
        i = fib(n-1)
        #$ omp end task
        #$ omp task shared(j) firstprivate(n)
        j = fib(n-2)
        #$ omp end task
        #$ omp taskwait
        return i + j
    #$ omp parallel
    #$ omp single
    m = fib(x)
    #$ omp end single
    #$ omp end parallel
    return m

def omp_simd(n : 'int'):
    from numpy import zeros
    func_result = 0
    arr = zeros(n, dtype=int)
    #$ omp parallel num_threads(4)
    #$ omp simd
    for i in range(0, n):
        arr[i] = i
    #$ omp end parallel
    for i in range(0, n):
        func_result = func_result + arr[i]
    return func_result

def omp_flush():
    from pyccel.stdlib.internal.openmp import omp_get_thread_num
    flag = 0
    #$ omp parallel num_threads(2)
    if omp_get_thread_num() == 0:
        #$ omp atomic update
        flag = flag + 1
    elif omp_get_thread_num() == 1:
        #$ omp flush(flag)
        while flag < 1:
            pass
            #$ omp flush(flag)
        #$ omp atomic update
        flag = flag + 1
    #$ omp end parallel
    return flag

def omp_barrier():
    from numpy import zeros
    arr = zeros(1000, dtype=int)
    func_result = 0
    #$ omp parallel num_threads(3)
    #$ omp for
    for i in range(0, 1000):
        arr[i] = i * 2

    #$ omp barrier
    #$ omp for reduction(+:func_result)
    for i in range(0, 1000):
        func_result = func_result + arr[i]
    #$ omp end parallel
    return func_result

def combined_for_simd():
    import numpy as np
    x = np.array([1,2,1,2,1,2,1,2])
    y = np.array([2,1,2,1,2,1,2,1])
    z = np.zeros(8, dtype = int)
    func_result = 0
    #$ omp parallel for simd
    for i in range(0, 8):
        z[i] = x[i] + y[i]

    for i in range(0, 8):
        func_result = func_result + z[i]
    return func_result

def omp_sections():
    n = 8
    sum1 = 0
    sum2 = 0
    sum3 = 0
    #$ omp parallel num_threads(2)
    #$ omp sections

    #$ omp section
    for i in range(0, int(n/3)):
        sum1 = sum1 + i
    #$ omp end section

    #$ omp section
    for i in range(0, int(n/2)):
        sum2 = sum2 + i
    #$ omp end section

    #$ omp section
    for i in range(0, n):
        sum3 = sum3 + i
    #$ omp end section
    #$ omp end sections

    #$ omp end parallel

    return (sum1 + sum2 + sum3)

def omp_long_line(long_variable_1_oiwed423rnoij21d4kojklm : 'int[:]', long_variable_2_oiwedqwrnoij2asxaxnjkna : 'int[:]', long_variable_3_oiweqxhnoijaqed34023423 : 'int[:]', long_variable_4_oiweaxaijaqedqd34023423 : 'int[:]', long_variable_5_oiwed423rnoic3242ewdx35 : 'int[:]'):
    func_result = 0
    n1     = long_variable_1_oiwed423rnoij21d4kojklm.shape[0]
    n2     = long_variable_2_oiwedqwrnoij2asxaxnjkna.shape[0]
    n3     = long_variable_3_oiweqxhnoijaqed34023423.shape[0]
    n4     = long_variable_4_oiweaxaijaqedqd34023423.shape[0]
    n5     = long_variable_5_oiwed423rnoic3242ewdx35.shape[0]

    #$ omp parallel private(i1, i2, i3, i4, i5) shared(long_variable_1_oiwed423rnoij21d4kojklm, long_variable_2_oiwedqwrnoij2asxaxnjkna, long_variable_3_oiweqxhnoijaqed34023423, long_variable_4_oiweaxaijaqedqd34023423, long_variable_5_oiwed423rnoic3242ewdx35, n1, n2, n3, n4, n5)

    #$ omp for reduction (+:func_result)
    for i1 in range(0, n1):
        func_result += long_variable_1_oiwed423rnoij21d4kojklm[i1]

    #$ omp for reduction (+:func_result)
    for i2 in range(0, n2):
        func_result += long_variable_2_oiwedqwrnoij2asxaxnjkna[i2]

    #$ omp for reduction (+:func_result)
    for i3 in range(0, n3):
        func_result += long_variable_3_oiweqxhnoijaqed34023423[i3]

    #$ omp for reduction (+:func_result)
    for i4 in range(0, n4):
        func_result += long_variable_4_oiweaxaijaqedqd34023423[i4]

    #$ omp for reduction (+:func_result)
    for i5 in range(0, n5):
        func_result += long_variable_5_oiwed423rnoic3242ewdx35[i5]

    #$ omp end parallel
    return func_result
