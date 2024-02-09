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
    def po():
        a = [1,3,45]
        return a.pop()
    re = epyccel(po, language = language)
    assert isinstance(po(), type(re()))
    assert po() == re()

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
    def po():
        a = [1j,3j,45j]
        return a.pop(1)
    re = epyccel(po, language = language)
    assert isinstance(po(), type(re()))
    assert po() == re()

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
    def po():
        a = [1j,3j,45j]
        return a.pop(-1)
    re = epyccel(po, language = language)
    assert isinstance(po(), type(re()))
    assert po() == re()

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
    def po():
        a = [1.7,2.7,45.0]
        a.pop()
        return a.pop(-1)
    re = epyccel(po, language = language)
    assert isinstance(po(), type(re()))
    assert po() == re()
