# pylint: disable=missing-function-docstring, missing-module-docstring/
# pylint: disable=wildcard-import
import multiprocessing
import os
import pytest
import numpy as np
import modules.openmp as openmp

from pyccel.epyccel import epyccel
#==============================================================================

def test_module_1(language):
    f1 = epyccel(openmp.f1, accelerator='openmp', language=language)
    set_num_threads = epyccel(openmp.set_num_threads, accelerator='openmp', language=language)
    get_num_threads = epyccel(openmp.get_num_threads, accelerator='openmp', language=language)
    get_max_threads = epyccel(openmp.get_max_threads, accelerator='openmp', language=language)
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

def test_modules_10(language):
    set_num_threads = epyccel(openmp.set_num_threads, accelerator='openmp', language=language)
    set_num_threads(1)
    f1 = epyccel(openmp.test_omp_get_ancestor_thread_num, accelerator='openmp', language=language)

    assert f1() == 0
    set_num_threads(4)

def test_module_2(language):
    f1 = epyccel(openmp.test_omp_number_of_procs, accelerator='openmp', language=language)
    assert f1() == multiprocessing.cpu_count()

def test_module_3(language):
    set_num_threads = epyccel(openmp.set_num_threads, accelerator='openmp', language=language)
    set_num_threads(4)
    f1 = epyccel(openmp.test_omp_in_parallel1, accelerator='openmp', language=language)
    f2 = epyccel(openmp.test_omp_in_parallel2, accelerator='openmp', language=language)

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
def test_modules_4(lang):
    f1 = epyccel(openmp.test_omp_set_get_dynamic, accelerator='openmp', language=lang)

    assert f1(1) == 1
    assert f1(0) == 0

@pytest.mark.parametrize( 'lang', (
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="omp_set_nested requires bool(kind=1) but C_BOOL has(kind=4)"),
            pytest.mark.fortran]
        )
    )
)
def test_modules_4_1(lang):
    f1 = epyccel(openmp.test_omp_set_get_nested, accelerator='openmp', language=lang)

    assert f1(1) == 1
    assert f1(0) == 0

def test_modules_5(language):
    f1 = epyccel(openmp.test_omp_get_cancellation, accelerator='openmp', language=language)

    cancel_var = os.environ.get('OMP_CANCELLATION')
    if cancel_var is not None:
        if cancel_var.lower() == 'true':
            assert f1() == 1
        else:
            assert f1() == 0
    else:
        assert f1() == 0

def test_modules_6(language):
    f1 = epyccel(openmp.test_omp_get_thread_limit, accelerator='openmp', language=language)
    #In order to test this function properly we must set the OMP_THREAD_LIMIT env var with the number of threads limit of the program
    #When the env var is not set, the number of threads limit is MAX INT
    assert f1() >= 0

def test_modules_9(language):
    f1 = epyccel(openmp.test_omp_get_active_level, accelerator='openmp', language=language)

    assert f1() == 1

def test_modules_7(language):
    f1 = epyccel(openmp.test_omp_get_set_max_active_levels, accelerator='openmp', language=language)

    max_active_level = 5
    #if the given max_active_level less than 0, omp_get_max_active_levels() gonna return (MAX_INT) as result
    assert f1(max_active_level) == max_active_level

def test_modules_8(language):
    f1 = epyccel(openmp.test_omp_get_level, accelerator='openmp', language=language)

    assert f1() == 2

def test_modules_11(language):
    set_num_threads = epyccel(openmp.set_num_threads, accelerator='openmp', language=language)
    set_num_threads(4)
    f1 = epyccel(openmp.test_omp_get_team_size, accelerator='openmp', language=language)

    assert f1() == 4
    set_num_threads(8)
    assert f1() == 8

@pytest.mark.xfail(reason = "Tasks not supported yet for openmp !")
def test_modules_12(language):
    f1 = epyccel(openmp.test_omp_in_final, accelerator='openmp', language=language)

    assert f1() == 1

def test_modules_13(language):
    f1 = epyccel(openmp.test_omp_get_proc_bind, accelerator='openmp', language=language)

    assert f1() >= 0

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="omp_get_num_devices and omp_get_default_device unrecognized in C !"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_modules_14_0(language):
    f1 = epyccel(openmp.test_omp_set_get_default_device, accelerator='openmp', language=language)
    f2 = epyccel(openmp.test_omp_get_num_devices, accelerator='openmp', language=language)

    assert f1(1) == 1
    assert f1(0) == 0
    assert f2() >= 0

# omp_get_initial_device give a compilation error on Travis (Linux and Windows), also Target construct not implemented yet !"
def test_modules_14_1(language):
    f3 = epyccel(openmp.test_omp_is_initial_device, accelerator='openmp', language=language)
    # f4 = epyccel(openmp.test_omp_get_initial_device, accelerator='openmp') #Target construct not implemented yet and need a non-host device to test the function

    assert f3() == 1

@pytest.mark.xfail(reason = "Teams not supported yet for openmp !")
def test_modules_15(language):
    f1 = epyccel(openmp.test_omp_get_num_teams, accelerator='openmp', language=language)
    f2 = epyccel(openmp.test_omp_get_team_num, accelerator='openmp', language=language)

    assert f1() == 2
    assert f2(0) == 0
    assert f2(1) == 1

@pytest.mark.xfail(reason = "Tasks not supported yet for openmp !")
def test_modules_16(language):
    f1 = epyccel(openmp.test_omp_get_max_task_priority, accelerator='openmp', language=language)

    assert f1() == 5

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="Numpy Arrays not implemented in C !"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_omp_matmul(language):
    f1 = epyccel(openmp.omp_matmul, accelerator='openmp', language=language)
    set_num_threads = epyccel(openmp.set_num_threads, accelerator='openmp', language=language)
    set_num_threads(4)
    from numpy import matmul
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
            pytest.mark.xfail(reason="Numpy Arrays not implemented in C !"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_omp_matmul_single(language):
    f1 = epyccel(openmp.omp_matmul_single, accelerator='openmp', language=language)
    set_num_threads = epyccel(openmp.set_num_threads, accelerator='openmp', language=language)
    set_num_threads(4)
    from numpy import matmul
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
            pytest.mark.xfail(reason="Numpy Arrays not implemented in C !"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_omp_matmul_2d_2d(language):
    f1 = epyccel(openmp.omp_matmul, accelerator='openmp', language=language)
    set_num_threads = epyccel(openmp.set_num_threads, accelerator='openmp', language=language)
    set_num_threads(4)
    from numpy import matmul
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

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="Numpy Arrays not implemented in C !"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_omp_arraysum(language):
    f1 = epyccel(openmp.omp_arraysum, accelerator='openmp', language=language)
    set_num_threads = epyccel(openmp.set_num_threads, accelerator='openmp', language=language)
    set_num_threads(4)
    from numpy import random
    x = random.randint(20, size=(5))

    assert f1(x) == np.sum(x)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="Numpy Arrays not implemented in C !"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_omp_arraysum_single(language):
    f1 = epyccel(openmp.omp_arraysum_single, accelerator='openmp', language=language)
    set_num_threads = epyccel(openmp.set_num_threads, accelerator='openmp', language=language)
    set_num_threads(2)
    from numpy import random
    x = random.randint(20, size=(10))

    assert f1(x) == np.sum(x)
