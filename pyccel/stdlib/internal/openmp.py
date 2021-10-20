# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
File providing functions to mimic OpenMP Runtime library routines to allow files to run
in pure python mode
"""

def omp_set_num_threads(num_threads : int):
    """
    The omp_set_num_threads routine affects the number of threads
    to be used for subsequent parallel regions that do not specify
    a num_threads clause, by setting the value of the first element
    of the nthreads-var ICV of the current task.

    Parameters
    ----------
    num_threads : int
    """

def omp_get_num_threads():
    """
    The omp_get_num_threads routine returns the number of threads
    in the current team.
    """
    return 1

def omp_get_max_threads():
    """
    The omp_get_max_threads routine returns an upper bound on the
    number of threads that could be used to form a new team if a
    parallel construct without a num_threads clause were encountered
    after execution returns from this routine.
    """
    return 1

def omp_get_thread_num():
    """
    The omp_get_thread_num routine returns the thread number,
    within the current team, of the calling thread
    """
    return 0

def omp_get_num_procs():
    """
    The omp_get_num_procs routine returns the number of processors
    available to the device.
    """
    return 1

def omp_in_parallel():
    """
    The omp_in_parallel routine returns true if the active-levels-var
    ICV is greater than zero; otherwise, it returns false
    """
    return False

def omp_set_dynamic(dynamic_threads : bool):
    """
    The omp_set_dynamic routine enables or disables dynamic
    adjustment of the number of threads available for the execution
    of subsequent parallel regions by setting the value of the
    dyn-var ICV

    Parameters
    ----------
     : bool
    """

def omp_get_dynamic():
    """
    The omp_get_dynamic routine returns the value of the dyn-var
    ICV, which determines whether dynamic adjustment of the number
    of threads is enabled or disabled.
    """
    return False

def omp_get_cancellation():
    """
    The omp_get_cancellation routine returns the value of the
    cancel-var ICV, which determines if cancellation is enabled
    or disabled.
    """
    return False

def omp_set_nested(nested : bool):
    """
    The deprecated omp_set_nested routine enables or disables
    nested parallelism by setting the max-active-levels-var ICV.

    Parameters
    ----------
    nested : bool
    """

def omp_get_nested():
    """
    The deprecated omp_get_nested routine returns whether nested
    parallelism is enabled or disabled, according to the value
    of the max-active-levels-var ICV.
    """
    return False

def omp_set_schedule(kind : int, chunk_size : int):
    """
    The omp_set_schedule routine affects the schedule that is
    applied when runtime is used as schedule kind, by setting
    the value of the run-sched-var ICV.

    Parameters
    ----------
    kind : int
    chunk_size : int
    """

def omp_get_schedule():
    """
    The omp_get_schedule routine returns the schedule that is
    applied when the runtime schedule is used.

    Results
    -------
    kind : int
    chunk_size : int
    """
    return 1,0

def omp_get_thread_limit():
    """
    The omp_get_thread_limit routine returns the maximum number
    of OpenMP threads available to participate in the current
    contention group.
    """
    return 1

def omp_set_max_active_levels(max_levels : int):
    """
    The omp_set_max_active_levels routine limits the number of
    nested active parallel regions on the device, by setting the
    max-active-levels-var ICV

    Parameters
    ----------
    max_levels : int
    """

def omp_get_max_active_levels():
    """
    The omp_get_max_active_levels routine returns the value of
    the max-active-levels-var ICV, which determines the maximum
    number of nested active parallel regions on the device.
    """
    return 1

def omp_get_level():
    """
    The omp_get_level routine returns the value of the levels-var ICV.
    """
    return 0

def omp_get_ancestor_thread_num(level : int):
    """
    The omp_get_ancestor_thread_num routine returns, for a given
    nested level of the current thread, the thread number of the
    ancestor of the current thread.

    Parameters
    ----------
    level : int
    """
    return -1

def omp_get_team_size(level : int):
    """
    The omp_get_team_size routine returns, for a given nested
    level of the current thread, the size of the thread team to
    which the ancestor or the current thread belongs.

    Parameters
    ----------
    level : int
    """
    return 1

def omp_get_active_level():
    """
    The omp_get_active_level routine returns the value of the
    active-level-vars ICV.
    """
    return 0

def omp_in_final():
    """
    The omp_in_final routine returns true if the routine is
    executed in a final task region; otherwise, it returns false.
    """
    return False

def omp_get_proc_bind():
    """
    The omp_get_proc_bind routine returns the thread affinity
    policy to be used for the subsequent nested parallel regions
    that do not specify a proc_bind clause.
    """
    return 0

def omp_get_num_places():
    """
    The omp_get_num_places routine returns the number of places
    available to the execution environment in the place list.
    """
    return 1

def omp_get_place_num_procs(place_num : int):
    """
    The omp_get_place_num_procs routine returns the number of
    processors available to the execution environment in the
    specified place.

    Parameters
    ----------
    place_num : int
    """
    return 1

def omp_get_place_proc_ids(place_num : int, ids : 'int[:]'):
    """
    The omp_get_place_proc_ids routine returns the numerical
    identifiers of the processors available to the execution
    environment in the specified place.

    Parameters
    ----------
    place_num : int
    ids : array of ints
            To be filled by the function
    """

def omp_get_place_num():
    """
    The omp_get_place_num routine returns the place number of
    the place to which the encountering thread is bound.
    """
    return -1

def omp_get_partition_num_places():
    """
    The omp_get_partition_num_places routine returns the number
    of places in the place partition of the innermost implicit task.
    """
    return 1

def omp_get_partition_place_nums(place_nums : 'int[:]'):
    """
    The omp_get_partition_place_nums routine returns the list of
    place numbers corresponding to the places in the
    place-partition-var ICV of the innermost implicit task.

    Parameters
    ----------
    place_nums : array of ints
            To be filled by the function
    """

def omp_set_default_device(device_num : int):
    """
    The omp_set_default_device routine controls the default
    target device by assigning the value of the
    default-device-var ICV.
    """

def omp_get_default_device():
    """
    The omp_get_default_device routine returns the default
    target device.
    """
    return 0

def omp_get_num_devices():
    """
    The omp_get_num_devices routine returns the number of
    target devices.
    """
    return 1

def omp_get_num_teams():
    """
    The omp_get_num_teams routine returns the number of initial
    teams in the current teams region.
    """
    return 1

def omp_get_team_num():
    """
    The omp_get_team_num routine returns the initial team number
    of the calling thread.
    """

def omp_is_initial_device():
    """
    The omp_is_initial_device routine returns true if the current
    task is executing on the host device; otherwise, it returns
    false.
    """
    return True

def omp_get_initial_device():
    """
    The omp_get_initial_device routine returns a device number
    that represents the host device.
    """
    return 0

def omp_get_max_task_priority():
    """
    The omp_get_max_task_priority routine returns the maximum
    value that can be specified in the priority clause.
    """
    return 0
