# pylint: disable=missing-function-docstring, missing-module-docstring/
import pytest

from pyccel.epyccel import epyccel
from pyccel.decorators import stack_array, types
from pyccel.errors.errors import Errors, PyccelSemanticError
from pyccel.errors.messages import (ARRAY_REALLOCATION,
                                    ARRAY_DEFINITION_IN_LOOP,
                                    INCOMPATIBLE_REDEFINITION_STACK_ARRAY,
                                    STACK_ARRAY_DEFINITION_IN_LOOP)

@pytest.fixture(params=[
    pytest.param('fortran', marks = pytest.mark.fortran),
    pytest.param('c'      , marks = [pytest.mark.c,
        pytest.mark.skip(message='Arrays not implemented in C')])
    ]
)
def language(request):
    return request.param

#==============================================================================
def test_no_reallocation(language):

    @stack_array('y')
    def f():
        import numpy as np

        x = np.zeros((2, 5), dtype=float)
        x = np.ones ((2, 5), dtype=float)

        y = np.zeros((2, 2, 1), dtype=int)
        y = np.ones ((2, 2, 1), dtype=int)

        return x.sum() + y.sum()

    # TODO: check that we don't get any Pyccel warnings
    g = epyccel(f, language=language)

    # Check result of pyccelized function
    assert f() == g()

#==============================================================================
def test_reallocation_heap(language):

    def f():
        import numpy as np
        x = np.zeros((3, 7), dtype=int)
        x = np.ones ((4, 5), dtype=int)
        return x.sum()

    # Initialize singleton that stores Pyccel errors
    errors = Errors()

    # TODO: check if we get the correct Pyccel warning
    g = epyccel(f, language=language)

    # Check result of pyccelized function
    assert f() == g()

    # Check that we got exactly 1 Pyccel warning
    assert errors.has_warnings()
    assert errors.num_messages() == 1

    # Check that we the warning is correct
    warning_info = [*errors.error_info_map.values()][0][0]
    assert warning_info.symbol  == 'x'
    assert warning_info.message == ARRAY_REALLOCATION

#==============================================================================
def test_reallocation_stack(language):

    @stack_array('x')
    def f():
        import numpy as np
        x = np.zeros((3, 7), dtype=int)
        x = np.ones ((4, 5), dtype=int)
        return x.sum()

    # Initialize singleton that stores Pyccel errors
    errors = Errors()

    # epyccel should raise an Exception
    with pytest.raises(PyccelSemanticError):
        epyccel(f, language=language)

    # Check that we got exactly 1 Pyccel error
    assert errors.has_errors()
    assert errors.num_messages() == 1

    # Check that we the error is correct
    error_info = [*errors.error_info_map.values()][0][0]
    assert error_info.symbol  == 'x'
    assert error_info.message == INCOMPATIBLE_REDEFINITION_STACK_ARRAY

#==============================================================================
def test_creation_in_loop_heap(language):

    def f():
        import numpy as np
        for i in range(3):
            x = np.ones(i, dtype=int)
        return x.sum()

    # Initialize singleton that stores Pyccel errors
    errors = Errors()

    # TODO: check if we get the correct Pyccel warning
    g = epyccel(f, language=language)

    # Check result of pyccelized function
    assert f() == g()

    # Check that we got exactly 1 Pyccel warning
    assert errors.has_warnings()
    assert errors.num_messages() == 1

    # Check that we the warning is correct
    warning_info = [*errors.error_info_map.values()][0][0]
    assert warning_info.symbol  == 'x'
    assert warning_info.message == ARRAY_DEFINITION_IN_LOOP

#==============================================================================
def test_creation_in_loop_stack(language):

    @stack_array('x')
    def f():
        import numpy as np
        for i in range(3):
            x = np.ones(i, dtype=int)
        return x.sum()

    # Initialize singleton that stores Pyccel errors
    errors = Errors()

    # epyccel should raise an Exception
    with pytest.raises(PyccelSemanticError):
        epyccel(f, language=language)

    # Check that we got exactly 1 Pyccel error
    assert errors.has_errors()
    assert errors.num_messages() == 1

    # Check that we the error is correct
    error_info = [*errors.error_info_map.values()][0][0]
    assert error_info.symbol  == 'x'
    assert error_info.message == STACK_ARRAY_DEFINITION_IN_LOOP

#==============================================================================
def test_creation_in_if_heap(language):

    @types('float')
    def f(c):
        import numpy as np
        if c > 0.5:
            x = np.ones(2, dtype=int)
        else:
            x = np.ones(7, dtype=int)
        return x.sum()

    # TODO: check if we get the correct Pyccel warning
    g = epyccel(f, language=language)

    # Check result of pyccelized function
    import numpy as np
    c = np.random.random()
    assert f(c) == g(c)

#==============================================================================
if __name__ == '__main__':

    for l in ['fortran']:

        test_no_reallocation(l)

        test_reallocation_heap(l)
        test_reallocation_stack(l)

        test_creation_in_loop_heap(l)
        test_creation_in_loop_stack(l)

        test_creation_in_if_heap(l)
