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

def test_omp_get_cancellation():
	from pyccel.stdlib.internal.openmp import omp_get_cancellation

	cancel_var = omp_get_cancellation()
	return cancel_var



@types ('int')
def test_omp_set_get_default_device(device_num):
	from pyccel.stdlib.internal.openmp import omp_get_default_device, omp_set_default_device

	omp_set_default_device(device_num)
	default_device = omp_get_default_device()
	return default_device

def test_omp_get_num_devices():
	from pyccel.stdlib.internal.openmp import omp_get_num_devices

	num_devices = omp_get_num_devices()
	return num_devices

def test_omp_get_num_teams():
	from pyccel.stdlib.internal.openmp import omp_get_num_teams

	num_teams = omp_get_num_teams()
	return num_teams

def test_omp_get_team_num():
	from pyccel.stdlib.internal.openmp import omp_get_team_num

	team_num = omp_get_team_num()
	return team_num

def test_omp_is_initial_device():
	from pyccel.stdlib.internal.openmp import omp_is_initial_device

	is_task_in_init_device = omp_is_initial_device()
	return is_task_in_init_device

def test_omp_get_initial_device():
	from pyccel.stdlib.internal.openmp import omp_get_initial_device

	host_device = omp_get_initial_device()
	return host_device

def test_omp_get_max_task_priority():
	from pyccel.stdlib.internal.openmp import omp_get_max_task_priority


	max_task_priority_var = omp_get_max_task_priority()
	return max_task_priority_var