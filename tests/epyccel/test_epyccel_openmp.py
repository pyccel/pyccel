# pylint: disable=wildcard-import

import pytest
import multiprocessing
import os
import numpy as np

from pyccel.epyccel import epyccel

#==============================================================================
import modules.openmp as openmp

mod = epyccel(openmp, accelerator='openmp')

def test_OMP_functions():
	mod.set_num_threads(4)
	assert mod.get_num_threads() == 4
	assert mod.f1(0) == 0
	assert mod.f1(1) == 1
	assert mod.f1(2) == 2
	assert mod.f1(3) == 3
	assert mod.f1(5) == -1

	assert mod.test_omp_number_of_procs() == multiprocessing.cpu_count()
	mod.set_num_threads(8)
	assert mod.get_num_threads() == 8
	cancel_var = os.environ.get('OMP_CANCELLATION')
	if cancel_var is not None:
		if cancel_var.lower() == 'true':
			assert mod.test_omp_get_cancellation() == 1
		else:
			assert mod.test_omp_get_cancellation() == 0
	else:
		assert mod.test_omp_get_cancellation() == 0

	assert mod.test_omp_in_parallel1() == 0
	assert mod.test_omp_in_parallel2() == 1

	assert mod.test_omp_set_get_dynamic(1) == 1
	assert mod.test_omp_set_get_dynamic(0) == 0

	#something wierd happening here, the return is a massive number
	assert mod.test_omp_get_thread_limit() >= 0

	max_active_level = 5
	#if the given max_active_level less than 0, omp_get_max_active_levels() gonna return (MAX_INT + (- max_active_level)) as result
	#example omp_get_max_active_levels(-1) will give 2147483647
	assert mod.test_omp_get_set_max_active_levels(max_active_level) == max_active_level

	assert mod.test_omp_get_level() >= 0

	assert mod.test_omp_get_ancestor_thread_num() >= 0

	assert mod.test_omp_get_team_size() > 0

	assert mod.test_omp_get_active_level() >= 0

	assert mod.test_omp_get_proc_bind() >= 0

	#OMP_PLACES (env var) should be set proply for this test
	#assert mod.test_omp_places() >= 0

	device_num = mod.test_omp_get_initial_device()
	mod.test_omp_set_get_default_device(device_num)

	assert mod.test_omp_get_num_devices() >= 0

	num_teams = mod.test_omp_get_num_teams()
	assert mod.test_omp_get_team_num() <= num_teams
	assert mod.test_omp_get_team_num() >= 0

	assert mod.test_omp_is_initial_device() == True

	assert mod.test_omp_get_max_task_priority() >= 0

	assert mod.f1(5) == 5

def test_array_real_2d_1d_matmul():
    mod.set_num_threads(4)
    from numpy import matmul
    A1 = np.ones([3, 2])
    A1[1,0] = 2
    A2 = np.copy(A1)
    x1 = np.ones([2, 1])
    x2 = np.copy(x1)
    y1 = np.zeros([3,1])
    y2 = np.zeros([3,1])
    mod.omp_matmul(A1, x1, y1)
    y2[:] = matmul(A2, x2)
    assert np.array_equal(y1, y2)
