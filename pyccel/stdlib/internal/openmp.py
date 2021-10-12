
def omp_set_num_threads(a : int):
    pass

def omp_get_num_threads():
    return 1

def omp_get_max_threads():
    return 1

def omp_get_thread_num():
    return 0

def omp_get_num_procs():
    return 1

def omp_in_parallel():
    return False

def omp_set_dynamic(b : bool):
    pass

def omp_get_dynamic():
    return False

def omp_get_cancellation():
    return False

def omp_set_nested(b : bool):
    pass

def omp_get_nested():
    return False

def omp_set_schedule(a : int, b : int):
    pass

def omp_get_schedule():
    return 1,0

def omp_get_thread_limit():
    return 1

def omp_set_max_active_levels(a : int):
    pass

def omp_get_max_active_levels():
    return 1

def omp_get_level():
    return 0

def omp_get_ancestor_thread_num(a : int):
    return -1

def omp_get_team_size(a : int):
    return 1

def omp_get_active_level():
    return 0

def omp_in_final():
    return False

def omp_get_proc_bind():
    return 0
