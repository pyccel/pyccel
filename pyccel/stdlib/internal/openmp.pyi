"""
Pyccel header for OpenMP.
OpenMP directives and Constructs are handled by the parser (see openmp.tx) and are parts of the Pyccel language.
We only list here what can not be described in the openmp grammar.
"""
#$ header metavar module_name = 'omp_lib'
#$ header metavar module_version = '4.5'
#$ header metavar save=True
#$ header metavar no_target=True
#$ header metavar external=False
from numpy import int32
from pyccel.decorators import low_level

# ............................................................
#            Runtime Library Routines for Fortran
# ............................................................

@low_level('omp_set_num_threads')
def omp_set_num_threads(anon_0 : 'int32') -> None:
    ...

@low_level('omp_get_num_threads')
def omp_get_num_threads() -> 'int32':
    ...

@low_level('omp_get_max_threads')
def omp_get_max_threads() -> 'int32':
    ...

@low_level('omp_get_thread_num')
def omp_get_thread_num() -> 'int32':
    ...

@low_level('omp_get_num_procs')
def omp_get_num_procs() -> 'int32':
    ...

@low_level('omp_in_parallel')
def omp_in_parallel() -> 'bool':
    ...

@low_level('omp_set_dynamic')
def omp_set_dynamic(anon_0 : 'bool') -> None:
    ...

@low_level('omp_get_dynamic')
def omp_get_dynamic() -> 'bool':
    ...

@low_level('omp_get_cancellation')
def omp_get_cancellation() -> 'bool':
    ...

@low_level('omp_set_nested')
def omp_set_nested(anon_0 : 'bool') -> None:
    ...

@low_level('omp_get_nested')
def omp_get_nested() -> 'bool':
    ...

@low_level('omp_set_schedule')
def omp_set_schedule(anon_0 : 'int32', anon_1 : 'int32') -> None:
    ...

@low_level('omp_get_schedule')
def omp_get_schedule() -> 'tuple[int32, int32]':
    ...

@low_level('omp_get_thread_limit')
def omp_get_thread_limit() -> 'int32':
    ...

@low_level('omp_set_max_active_levels')
def omp_set_max_active_levels(anon_0 : 'int32') -> None:
    ...

@low_level('omp_get_max_active_levels')
def omp_get_max_active_levels() -> 'int32':
    ...

@low_level('omp_get_level')
def omp_get_level() -> 'int32':
    ...

@low_level('omp_get_ancestor_thread_num')
def omp_get_ancestor_thread_num(anon_0 : 'int32') -> 'int32':
    ...

@low_level('omp_get_team_size')
def omp_get_team_size(anon_0 : 'int32') -> 'int32':
    ...

@low_level('omp_get_active_level')
def omp_get_active_level() -> 'int32':
    ...

@low_level('omp_in_final')
def omp_in_final() -> 'bool':
    ...

@low_level('omp_get_proc_bind')
def omp_get_proc_bind() -> 'int32':
    ...

@low_level('omp_get_num_places')
def omp_get_num_places() -> 'int32':
    ...

@low_level('omp_get_place_num_procs')
def omp_get_place_num_procs(anon_0 : 'int32') -> 'int32':
    ...

@low_level('omp_get_place_proc_ids')
def omp_get_place_proc_ids(anon_0 : 'int32', anon_1 : 'int32[:]') -> None:
    ...

@low_level('omp_get_place_num')
def omp_get_place_num() -> 'int32':
    ...

@low_level('omp_get_partition_num_places')
def omp_get_partition_num_places() -> 'int32':
    ...

@low_level('omp_get_partition_place_nums')
def omp_get_partition_place_nums(anon_0 : 'int32[:]') -> None:
    ...

@low_level('omp_set_default_device')
def omp_set_default_device(anon_0 : 'int32') -> None:
    ...

@low_level('omp_get_default_device')
def omp_get_default_device() -> 'int32':
    ...

@low_level('omp_get_num_devices')
def omp_get_num_devices() -> 'int32':
    ...

@low_level('omp_get_num_teams')
def omp_get_num_teams() -> 'int32':
    ...

@low_level('omp_get_team_num')
def omp_get_team_num() -> 'int32':
    ...

@low_level('omp_is_initial_device')
def omp_is_initial_device() -> 'bool':
    ...

@low_level('omp_get_initial_device')
def omp_get_initial_device() -> 'int32':
    ...

@low_level('omp_get_max_task_priority')
def omp_get_max_task_priority() -> 'int32':
    ...

