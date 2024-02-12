# pylint: disable=missing-function-docstring, missing-module-docstring
import pytest
from  pyccel.epyccel import epyccel

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="Function in function not implemented in fortran"),
            pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="Function in function not implemented in C"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)

def test_pop_last_element(language) :
    def pop_last_element():
        a = [1,3,45]
        return a.pop()
    epyc_last_element = epyccel(pop_last_element, language = language)
    pyccel_result = epyc_last_element()
    python_result = pop_last_element()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="method pop for list not implemented in fortran"),
            pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="method pop for list not implemented in C"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)

def test_pop_specific_index(language) :
    def pop_specific_index():
        a = [1j,3j,45j]
        return a.pop(1)
    epyc_specific_index = epyccel(pop_specific_index, language = language)
    python_result = pop_specific_index()
    pyccel_result = epyc_specific_index()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="method pop for list not implemented in fortran"),
            pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="method pop for list not implemented in C"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)

def test_pop_negative_index(language) :
    def pop_negative_index():
        a = [1j,3j,45j]
        return a.pop(-1)
    epyc_negative_index = epyccel(pop_negative_index, language = language)
    python_result = pop_negative_index()
    pyccel_result = epyc_negative_index()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="method pop for list not implemented in fortran"),
            pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="method pop for list not implemented in C"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)

def test_pop_2(language) :
    def pop_2():
        a = [1.7,2.7,45.0]
        a.pop()
        return a.pop(-1)
    pop_2_epyc = epyccel(pop_2, language = language)
    python_result = pop_2()
    pyccel_result = pop_2_epyc()
    assert isinstance(python_result, type(pyccel_result))
    assert python_result == pyccel_result
