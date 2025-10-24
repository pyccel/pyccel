# pylint: disable=missing-function-docstring, missing-module-docstring
import multiprocessing
import os
import sys
import pytest
import numpy as np
import modules.openmp as openmp

from numpy import random
from numpy import matmul
from pyccel import epyccel

#==============================================================================

@pytest.mark.external
def test_directive_in_else(language):
    f1 = epyccel(openmp.directive_in_else, flags = '-Wall', openmp=True, language=language)
    assert f1(0)  == 0
    assert f1(15) == 15
    assert f1(32) == 496
    assert f1(40) == 780

@pytest.mark.parametrize( 'language', (
    pytest.param('fortran', marks = pytest.mark.fortran),
    pytest.param('c'      , marks = pytest.mark.c),
    pytest.param("python", marks = [
        pytest.mark.skip(reason="No parallelisation leads to different results"),
        pytest.mark.python])
    )
)
@pytest.mark.external
def test_module_1(language):
    f1 = epyccel(openmp.f1, flags = '-Wall', openmp=True, language=language)
    set_num_threads = epyccel(openmp.set_num_threads, flags = '-Wall', openmp=True, language=language)
    get_num_threads = epyccel(openmp.get_num_threads, flags = '-Wall', openmp=True, language=language)
    get_max_threads = epyccel(openmp.get_max_threads, flags = '-Wall', openmp=True, language=language)
    set_num_threads(4)
    assert get_max_threads() == 4
    assert get_num_threads() == 4
    assert f1(0) == 0
    assert f1(1) == 1
    assert f1(2) == 2
    assert f1(3) == 3
    assert f1(5) == -1

    set_num_threads(8)
    assert get_num_threads() == 8
    assert f1(5) == 5
    set_num_threads(4)

@pytest.mark.parametrize( 'language', (
    pytest.param('fortran', marks = pytest.mark.fortran),
    pytest.param('c'      , marks = pytest.mark.c),
    pytest.param("python", marks = [
        pytest.mark.skip(reason="No parallelisation leads to different results"),
        pytest.mark.python])
    )
)
@pytest.mark.external
def test_modules_10(language):
    set_num_threads = epyccel(openmp.set_num_threads, flags = '-Wall', openmp=True, language=language)
    set_num_threads(1)
    f1 = epyccel(openmp.test_omp_get_ancestor_thread_num, flags = '-Wall', openmp=True, language=language)

    assert f1() == 0
    set_num_threads(4)

@pytest.mark.parametrize( 'language', (
    pytest.param('fortran', marks = pytest.mark.fortran),
    pytest.param('c'      , marks = pytest.mark.c),
    pytest.param("python", marks = [
        pytest.mark.skip(reason="No parallelisation leads to different results"),
        pytest.mark.python])
    )
)
@pytest.mark.external
def test_module_2(language):
    f1 = epyccel(openmp.test_omp_number_of_procs, flags = '-Wall', openmp=True, language=language)
    assert f1() == multiprocessing.cpu_count()

@pytest.mark.parametrize( 'language', (
    pytest.param('fortran', marks = pytest.mark.fortran),
    pytest.param('c'      , marks = pytest.mark.c),
    pytest.param("python", marks = [
        pytest.mark.skip(reason="No parallelisation leads to different results"),
        pytest.mark.python])
    )
)
@pytest.mark.external
def test_module_3(language):
    set_num_threads = epyccel(openmp.set_num_threads, flags = '-Wall', openmp=True, language=language)
    set_num_threads(4)
    f1 = epyccel(openmp.test_omp_in_parallel1, flags = '-Wall', openmp=True, language=language)
    f2 = epyccel(openmp.test_omp_in_parallel2, flags = '-Wall', openmp=True, language=language)

    assert f1() == 0
    assert f2() == 1

@pytest.mark.parametrize( 'lang', (
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="omp_set_dynamic requires bool(kind=1) but C_BOOL has(kind=4)"),
            pytest.mark.fortran]
        )
    )
)
@pytest.mark.external
def test_modules_4(lang):
    f1 = epyccel(openmp.test_omp_set_get_dynamic, flags = '-Wall', openmp=True, language=lang)

    assert f1(True) == 1
    assert f1(False) == 0

@pytest.mark.parametrize( 'lang', (
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="omp_set_nested requires bool(kind=1) but C_BOOL has(kind=4)"),
            pytest.mark.fortran]
        )
    )
)
@pytest.mark.external
def test_modules_4_1(lang):
    f1 = epyccel(openmp.test_omp_set_get_nested, flags = '-Wall', openmp=True, language=lang)

    assert f1(True) == 1
    assert f1(False) == 0

@pytest.mark.external
def test_modules_5(language):
    f1 = epyccel(openmp.test_omp_get_cancellation, flags = '-Wall', openmp=True, language=language)

    cancel_var = os.environ.get('OMP_CANCELLATION')
    if cancel_var is not None:
        if cancel_var.lower() == 'true':
            assert f1() == 1
        else:
            assert f1() == 0
    else:
        assert f1() == 0

@pytest.mark.external
def test_modules_6(language):
    f1 = epyccel(openmp.test_omp_get_thread_limit, flags = '-Wall', openmp=True, language=language)
    #In order to test this function properly we must set the OMP_THREAD_LIMIT env var with the number of threads limit of the program
    #When the env var is not set, the number of threads limit is MAX INT
    assert f1() >= 0

@pytest.mark.parametrize( 'language', (
    pytest.param('fortran', marks = pytest.mark.fortran),
    pytest.param('c'      , marks = pytest.mark.c),
    pytest.param("python", marks = [
        pytest.mark.skip(reason="No parallelisation leads to different results"),
        pytest.mark.python])
    )
)
@pytest.mark.external
def test_modules_9(language):
    f1 = epyccel(openmp.test_omp_get_active_level, flags = '-Wall', openmp=True, language=language)

    assert f1() == 1

@pytest.mark.parametrize( 'language', (
    pytest.param('fortran', marks = pytest.mark.fortran),
    pytest.param('c'      , marks = pytest.mark.c),
    pytest.param("python", marks = [
        pytest.mark.skip(reason="No parallelisation leads to different results"),
        pytest.mark.python])
    )
)
@pytest.mark.external
def test_modules_7(language):
    f1 = epyccel(openmp.test_omp_get_set_max_active_levels, flags = '-Wall', openmp=True, language=language)

    max_active_level = 5
    #if the given max_active_level less than 0, omp_get_max_active_levels() gonna return (MAX_INT) as result
    assert f1(max_active_level) == max_active_level

@pytest.mark.parametrize( 'language', (
    pytest.param('fortran', marks = pytest.mark.fortran),
    pytest.param('c'      , marks = pytest.mark.c),
    pytest.param("python", marks = [
        pytest.mark.skip(reason="No parallelisation leads to different results"),
        pytest.mark.python])
    )
)
@pytest.mark.external
def test_modules_8(language):
    f1 = epyccel(openmp.test_omp_get_level, flags = '-Wall', openmp=True, language=language)

    assert f1() == 2

@pytest.mark.parametrize( 'language', (
    pytest.param('fortran', marks = pytest.mark.fortran),
    pytest.param('c'      , marks = pytest.mark.c),
    pytest.param("python", marks = [
        pytest.mark.skip(reason="No parallelisation leads to different results"),
        pytest.mark.python])
    )
)
@pytest.mark.external
def test_modules_11(language):
    set_num_threads = epyccel(openmp.set_num_threads, flags = '-Wall', openmp=True, language=language)
    set_num_threads(4)
    f1 = epyccel(openmp.test_omp_get_team_size, flags = '-Wall', openmp=True, language=language)

    assert f1() == 4
    set_num_threads(8)
    assert f1() == 8

@pytest.mark.xfail(reason = "arithmetic expression not managed yet inside a clause !")
@pytest.mark.external
def test_modules_12(language):
    f1 = epyccel(openmp.test_omp_in_final, flags = '-Wall', openmp=True, language=language)

    assert f1() == 1

@pytest.mark.external
def test_modules_13(language):
    f1 = epyccel(openmp.test_omp_get_proc_bind, flags = '-Wall', openmp=True, language=language)

    assert f1() >= 0

#@pytest.mark.parametrize( 'language', [
#        pytest.param("c", marks = [
#            pytest.mark.xfail(sys.platform == 'darwin', reason="omp_get_num_devices and omp_get_default_device unrecognized in C !"),
#            pytest.mark.c]),
#        pytest.param("fortran", marks = pytest.mark.fortran)
#    ]
#)
@pytest.mark.skip("Compiling is not fully managed for GPU commands. See #798")
@pytest.mark.external
def test_modules_14_0(language):
    f1 = epyccel(openmp.test_omp_set_get_default_device, flags = '-Wall', openmp=True, language=language)
    f2 = epyccel(openmp.test_omp_get_num_devices, flags = '-Wall', openmp=True, language=language)

    assert f1(1) == 1
    assert f1(2) == 2
    assert f2() >= 0

@pytest.mark.skip("Compiling is not fully managed for GPU commands. See #798")
@pytest.mark.external
def test_modules_14_1(language):
    f3 = epyccel(openmp.test_omp_is_initial_device, flags = '-Wall', openmp=True, language=language)
    f4 = epyccel(openmp.test_omp_get_initial_device, flags = '-Wall', openmp=True, language=language) #Needs a non-host device to test the function properly

    assert f3() == 1
    assert f4() == 0

#@pytest.mark.parametrize( 'language', [
#        pytest.param("c", marks = [
#            pytest.mark.xfail(reason="omp_get_team_num() return a wrong result!"),
#            pytest.mark.c]),
#        pytest.param("fortran", marks = [
#            pytest.mark.xfail(reason="Compilation fails on github action"),
#            pytest.mark.fortran])
#
#    ]
#)
@pytest.mark.skip("Compiling is not fully managed for GPU commands. See #798")
@pytest.mark.external
def test_modules_15(language):
    f1 = epyccel(openmp.test_omp_get_team_num, flags = '-Wall', openmp=True, language=language)

    assert f1(0) == 0
    assert f1(1) == 1

#@pytest.mark.parametrize( 'language', [
#        pytest.param("c", marks = [
#            pytest.mark.xfail(reason="omp_get_num_teams() return a wrong result!"),
#            pytest.mark.c]),
#        pytest.param("fortran", marks = [
#            pytest.mark.xfail(reason="Compilation fails on github action"),
#            pytest.mark.fortran])
#    ]
#)
@pytest.mark.skip("Compiling is not fully managed for GPU commands. See #798")
@pytest.mark.external
def test_modules_15_1(language):
    f1 = epyccel(openmp.test_omp_get_num_teams, flags = '-Wall', openmp=True, language=language)

    assert f1() == 2

@pytest.mark.external
def test_modules_16(language):
    f1 = epyccel(openmp.test_omp_get_max_task_priority, flags = '-Wall', openmp=True, language=language)

    assert f1() == 0 # omp_get_max_task_priority() return always 0

@pytest.mark.external
def test_omp_matmul(language):
    f1 = epyccel(openmp.omp_matmul, flags = '-Wall', openmp=True, language=language)
    set_num_threads = epyccel(openmp.set_num_threads, flags = '-Wall', openmp=True, language=language)
    set_num_threads(4)
    A1 = np.ones([3, 2])
    A1[1,0] = 2
    A2 = np.copy(A1)
    x1 = np.ones([2, 1])
    x2 = np.copy(x1)
    y1 = np.zeros([3,1])
    y2 = np.zeros([3,1])
    f1(A1, x1, y1)
    y2[:] = matmul(A2, x2)

    assert np.array_equal(y1, y2)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="Numpy matmul not implemented in C !"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
@pytest.mark.external
def test_omp_matmul_single(language):
    f1 = epyccel(openmp.omp_matmul_single, flags = '-Wall', openmp=True, language=language)
    set_num_threads = epyccel(openmp.set_num_threads, flags = '-Wall', openmp=True, language=language)
    set_num_threads(4)
    A1 = np.ones([3, 2])
    A1[1,0] = 2
    A2 = np.copy(A1)
    x1 = np.ones([2, 1])
    x2 = np.copy(x1)
    y1 = np.zeros([3,1])
    y2 = np.zeros([3,1])
    f1(A1, x1, y1)
    y2[:] = matmul(A2, x2)

    assert np.array_equal(y1, y2)

@pytest.mark.external
def test_omp_matmul_2d_2d(language):
    f1 = epyccel(openmp.omp_matmul, flags = '-Wall', openmp=True, language=language)
    set_num_threads = epyccel(openmp.set_num_threads, flags = '-Wall', openmp=True, language=language)
    set_num_threads(4)
    A1 = np.ones([3, 2])
    A1[1,0] = 2
    A2 = np.copy(A1)
    x1 = np.ones([2, 3])
    x2 = np.copy(x1)
    y1 = np.zeros([3,3])
    y2 = np.zeros([3,3])
    f1(A1, x1, y1)
    y2[:] = matmul(A2, x2)

    assert np.array_equal(y1, y2)


@pytest.mark.external
def test_omp_nowait(language):
    f1 = epyccel(openmp.omp_nowait, flags = '-Wall', openmp=True, language=language)
    set_num_threads = epyccel(openmp.set_num_threads, flags = '-Wall', openmp=True, language=language)
    set_num_threads(4)
    x = np.array(random.randint(20, size=(1000)), dtype=int)
    y = np.zeros((1000,), dtype=int)
    z = np.zeros((1000,))
    f1(x, y, z)

    assert np.array_equal(y, x*2)
    assert np.array_equal(z, x/2)

@pytest.mark.external
def test_omp_arraysum(language):
    f1 = epyccel(openmp.omp_arraysum, flags = '-Wall', openmp=True, language=language)
    set_num_threads = epyccel(openmp.set_num_threads, flags = '-Wall', openmp=True, language=language)
    set_num_threads(4)
    x = np.array(random.randint(20, size=(5)), dtype=int)

    assert f1(x) == np.sum(x)

@pytest.mark.external
def test_omp_arraysum_combined(language):
    f1 = epyccel(openmp.omp_arraysum_combined, flags = '-Wall', openmp=True, language=language)
    set_num_threads = epyccel(openmp.set_num_threads, flags = '-Wall', openmp=True, language=language)
    set_num_threads(4)
    x = np.array(random.randint(20, size=(5)), dtype=int)

    assert f1(x) == np.sum(x)

@pytest.mark.external
def test_omp_range_sum_critical(language):
    f1 = epyccel(openmp.omp_range_sum_critical, flags = '-Wall', openmp=True, language=language)

    for _ in range(0, 4):
        x = random.randint(10, 1000)
        assert f1(x) == openmp.omp_range_sum_critical(x)

@pytest.mark.external
def test_omp_arraysum_single(language):
    f1 = epyccel(openmp.omp_arraysum_single, flags = '-Wall', openmp=True, language=language)
    set_num_threads = epyccel(openmp.set_num_threads, flags = '-Wall', openmp=True, language=language)
    set_num_threads(2)
    x = np.array(random.randint(20, size=(10)), dtype=int)

    assert f1(x) == np.sum(x)

@pytest.mark.external
def test_omp_master(language):
    f1 = epyccel(openmp.omp_master, flags = '-Wall', openmp=True, language=language)
    assert f1() == openmp.omp_master()

@pytest.mark.skipif_by_language(os.environ.get('PYCCEL_DEFAULT_COMPILER', 'GNU') == 'llvm', reason="flang error: not yet implemented: omp.taskloop", language='fortran')
@pytest.mark.parametrize( 'language', [
            pytest.param("python", marks = [
            pytest.mark.xfail(reason="The result of this test depend on threads, so in python we get different result because we don't use threads."),
            pytest.mark.python]),
            pytest.param("fortran", marks = pytest.mark.fortran),
            pytest.param("c", marks = pytest.mark.c)
    ]
)
@pytest.mark.external
def test_omp_taskloop(language):
    f1 = epyccel(openmp.omp_taskloop, flags = '-Wall', openmp=True, language=language)

    for _ in range(0, 4):
        x = random.randint(1, 4)
        result = 0
        for _ in range(0, x * 10):
            result = result + 1
        assert result == f1(x)

@pytest.mark.parametrize( 'language', [
            pytest.param("c", marks = [
            pytest.mark.xfail(reason="Nested functions not handled for C !", run=False),
            pytest.mark.c]),
            pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
@pytest.mark.external
def test_omp_tasks(language):
    f1 = epyccel(openmp.omp_tasks, flags = '-Wall', openmp=True, language=language)

    for _ in range(0, 4):
        x = random.randint(10, 20)
        assert openmp.omp_tasks(x) == f1(x)

@pytest.mark.external
def test_omp_simd(language):
    f1 = epyccel(openmp.omp_simd, flags = '-Wall', openmp=True, language=language)
    assert openmp.omp_simd(1337) == f1(1337)

@pytest.mark.external
def test_omp_long_line(language):
    f1 = epyccel(openmp.omp_long_line, flags = '-Wall', openmp=True, language=language)
    set_num_threads = epyccel(openmp.set_num_threads, flags = '-Wall', openmp=True, language=language)
    set_num_threads(4)
    x1 = np.array(random.randint(20, size=(5)), dtype=int)
    x2 = np.array(random.randint(20, size=(5)), dtype=int)
    x3 = np.array(random.randint(20, size=(5)), dtype=int)
    x4 = np.array(random.randint(20, size=(5)), dtype=int)
    x5 = np.array(random.randint(20, size=(5)), dtype=int)

    assert f1(x1,x2,x3,x4,x5) == np.sum(x1+x2+x3+x4+x5)

@pytest.mark.parametrize( 'language', (
    pytest.param('fortran', marks = pytest.mark.fortran),
    pytest.param('c'      , marks = pytest.mark.c),
    pytest.param("python", marks = [
        pytest.mark.skip(reason="No parallelisation leads to different results"),
        pytest.mark.python])
    )
)
@pytest.mark.external
def test_omp_flush(language):
    f1 = epyccel(openmp.omp_flush, flags = '-Wall', openmp=True, language=language)
    assert 2 == f1()

@pytest.mark.external
def test_omp_barrier(language):
    f1 = epyccel(openmp.omp_barrier, flags = '-Wall', openmp=True, language=language)
    f2 = openmp.omp_barrier
    assert f1() == f2()

@pytest.mark.external
def test_combined_for_simd(language):
    # Intel compiler has a bug in debug mode
    f1 = epyccel(openmp.combined_for_simd, flags = '-Wall', openmp=True, language=language, debug=False)
    f2 = openmp.combined_for_simd
    assert f1() == f2()

@pytest.mark.external
def test_omp_sections(language):
    f1 = epyccel(openmp.omp_sections, flags = '-Wall', openmp=True, language=language)
    f2 = openmp.omp_sections
    assert f1() == f2()

@pytest.mark.skipif(sys.platform == 'win32', reason="Type mismatch warning is an error on Windows")
@pytest.mark.external
def test_omp_get_set_schedule(language):
    set_num_threads = epyccel(openmp.set_num_threads, flags = '-Wall', openmp=True, language=language)
    set_num_threads(4)
    # Don't set -Wall as get_schedule should use enum type omp_sched_t
    f1 = epyccel(openmp.test_omp_get_set_schedule, openmp=True, language=language)

    result = f1()
    if language == 'python':
        assert result == 0
    else:
        assert result == 16*3

@pytest.mark.parametrize( 'language', (
    pytest.param('fortran', marks = pytest.mark.fortran),
    pytest.param("c", marks = [
        pytest.mark.skip(reason="Min and max are not implemented in C"),
        pytest.mark.c]),
    pytest.param("python", marks = pytest.mark.python)
    )
)
@pytest.mark.external
def test_nowait_schedule(language):
    set_num_threads = epyccel(openmp.set_num_threads, flags = '-Wall', openmp=True, language=language)
    get_num_threads = epyccel(openmp.get_num_threads, flags = '-Wall', openmp=True, language=language)
    f1 = epyccel(openmp.test_nowait_schedule, flags = '-Wall', openmp=True, language=language)

    set_num_threads(4)
    nthreads = get_num_threads()
    if language != 'python':
        assert nthreads == 4

    n = 200
    results = f1(n)
    min_vals = results[:nthreads]
    max_vals = results[4:4+nthreads]
    for i,m in enumerate(min_vals):
        assert m == i*n/nthreads
    for i,m in enumerate(max_vals):
        assert m == (i+1)*n/nthreads-1
