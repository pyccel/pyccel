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

def test_omp_get_set_schedule():
	from pyccel.stdlib.internal.openmp import omp_get_schedule, omp_set_schedule

	#omp_set_schedule(0, 0) #ERROR at Fortran compilation stage
	#schedule_kind, chunk_size = omp_get_schedule() #ERROR at Fortran compilation stage
	return 0




def test_omp_get_thread_limit():
	from pyccel.stdlib.internal.openmp import omp_get_thread_limit

	maximum_threads_available = omp_get_thread_limit()
	return maximum_threads_available


@types ('int')
def test_omp_get_set_max_active_levels(max_active_levels):
	from pyccel.stdlib.internal.openmp import omp_get_max_active_levels, omp_set_max_active_levels

	omp_set_max_active_levels(max_active_levels)
	max_active_levels_var = omp_get_max_active_levels()
	return max_active_levels_var


def test_omp_get_level():
	from pyccel.stdlib.internal.openmp import omp_get_active_level

	nested_parallel_regions = omp_get_active_level()
	return nested_parallel_regions


def test_omp_get_ancestor_thread_num():
	from pyccel.stdlib.internal.openmp import omp_get_ancestor_thread_num, omp_get_active_level

	active_level = omp_get_active_level()
	ancestor_thread = omp_get_ancestor_thread_num(active_level)
	return ancestor_thread

def test_omp_get_team_size():
	from pyccel.stdlib.internal.openmp import omp_get_team_size, omp_get_active_level

	active_level = omp_get_active_level()
	team_size = omp_get_team_size(active_level)
	return team_size

def test_omp_get_active_level():
	from pyccel.stdlib.internal.openmp import omp_get_active_level

	active_level_vars = omp_get_active_level()
	return active_level_vars

def test_omp_in_final():
	from pyccel.stdlib.internal.openmp import omp_in_final

	task_region = omp_in_final()
	return task_region

def test_omp_get_proc_bind():
	from pyccel.stdlib.internal.openmp import omp_get_proc_bind

	bind_var = omp_get_proc_bind()
	return bind_var

def test_omp_places():
	from pyccel.stdlib.internal.openmp import omp_get_partition_num_places, omp_get_partition_place_nums, omp_get_place_num, omp_get_place_proc_ids, omp_get_place_num_procs, omp_get_num_places

	partition_num_places = omp_get_partition_num_places()
	#partition_places_num =  omp_get_partition_place_nums() #ERROR at Fortran compilation stage
	place_num = omp_get_place_num()
	if place_num < 0:
		return -1
	#place_num, ids = omp_get_place_proc_ids() ERROR at Fortran compilation stage
	procs = omp_get_place_num_procs(place_num)
	num_places = omp_get_num_places()

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

@types('real[:,:], real[:,:], real[:,:]')
def omp_matmul(A, x, out):
	#$ omp parallel shared(A,x,out) private(i,j,k)
	#$ omp do
	for i in range(len(A)):
		for j in range(len(x[0])):
			for k in range(len(x)):
				out[i][j] += A[i][k] * x[k][j]
	#$ omp end do
	#$ omp end parallel

@types('real[:,:], real[:,:], real[:,:]')
def omp_matmul_single(A, x, out):
  from numpy import matmul
  #$ omp parallel
  #$ omp single
  out[:] = matmul(A, x)
  #$ omp end single
  #$ omp end parallel

@types('int[:]')
def omp_arraysum(x):
	result = 0
	#$ omp parallel private(i) reduction(+: result)
	#$ omp do
	for i in range(len(x)):
		result += x[i]
	#$ omp end do
	#$ omp end parallel
	return result

@types('int[:]')
def omp_arraysum_single(x):
  from numpy import sum
  result = 0
  #$ omp parallel
  #$ omp single
  result = sum(x)
  #$ omp end single
  #$ omp end parallel
  return result
