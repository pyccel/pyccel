import pytest
import multiprocessing
import os
import numpy as np

from pyccel.epyccel import epyccel

#==============================================================================
def test_module_1():
	import modules.openmp as openmp

	mod = epyccel(openmp, accelerator='openmp')
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

	device_num = mod.test_omp_get_initial_device()
	mod.test_omp_set_get_default_device(device_num)

	assert mod.test_omp_get_num_devices() >= 0

	num_teams = mod.test_omp_get_num_teams()
	assert mod.test_omp_get_team_num() <= num_teams
	assert mod.test_omp_get_team_num() >= 0

	assert mod.test_omp_is_initial_device() == True

	assert mod.test_omp_get_max_task_priority() >= 0

	assert mod.f1(5) == 5
