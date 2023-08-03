# pylint: disable=missing-function-docstring, missing-module-docstring
import os
import sys
import warnings
import pytest

from pyccel.epyccel import epyccel
from pyccel.decorators import stack_array, types
from pyccel.errors.errors import Errors, PyccelSemanticError
from pyccel.errors.messages import (ARRAY_REALLOCATION,
                                    ARRAY_DEFINITION_IN_LOOP,
                                    INCOMPATIBLE_REDEFINITION_STACK_ARRAY,
                                    STACK_ARRAY_DEFINITION_IN_LOOP,
                                    ASSIGN_ARRAYS_ONE_ANOTHER, ARRAY_ALREADY_IN_USE,
                                    STACK_ARRAY_UNKNOWN_SHAPE)

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

    # Check that the warning is correct
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

    # Check that the error is correct
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

    # Check that the warning is correct
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

    # Check that we got exactly 2 Pyccel errors
    assert errors.has_errors()
    if errors.mode == 'developer':
        assert errors.num_messages() == 1
    else:
        assert errors.num_messages() == 2

    # Check that the errors are correct
    error_info_list = [*errors.error_info_map.values()][0]
    error_info = error_info_list[0]
    assert error_info.symbol  == 'x'
    assert error_info.message == STACK_ARRAY_UNKNOWN_SHAPE
    error_info = error_info_list[1]
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
def test_Reassign_to_Target():

    def f():
        import numpy as np
        x = np.zeros((3, 7), dtype=int)
        c = x
        x = np.ones ((4, 5), dtype=int)
        return c.sum()

     # Initialize singleton that stores Pyccel errors
    errors = Errors()

    # epyccel should raise an Exception
    with pytest.raises(PyccelSemanticError):
        epyccel(f)

    # Check that we got exactly 1 Pyccel error
    assert errors.has_errors() == 1
    assert errors.num_messages() == 1

    # Check that the error is correct
    error_info = [*errors.error_info_map.values()][0][0]
    assert error_info.symbol  == 'x'
    assert error_info.message == ARRAY_ALREADY_IN_USE

#==============================================================================

def test_Assign_Between_Allocatables():

    def f():
        import numpy as np
        x = np.zeros((3, 7), dtype=int)
        y = np.ones ((4, 5), dtype=int)
        x = y
        x[0][0] = 1
        return y.sum()

     # Initialize singleton that stores Pyccel errors
    errors = Errors()

    # epyccel should raise an Exception
    with pytest.raises(PyccelSemanticError):
        epyccel(f)

    # Check that we got exactly 1 Pyccel error
    assert errors.has_errors() == 1
    assert errors.num_messages() == 1

    # Check that the error is correct
    error_info = [*errors.error_info_map.values()][0][0]
    assert str(error_info.symbol)  == 'x'
    assert error_info.message == ASSIGN_ARRAYS_ONE_ANOTHER

#==============================================================================

def test_Assign_after_If():

    def f(b : bool):
        import numpy as np
        if b:
            x = np.zeros(3, dtype=int)
        else:
            x = np.zeros(4, dtype=int)
        n = x.shape[0]
        x = np.ones(3, dtype=int)
        m = x.shape[0]
        return n,m

     # Initialize singleton that stores Pyccel errors
    errors = Errors()

    # epyccel should raise an Exception
    f2 = epyccel(f)

    # Check that we got exactly 1 Pyccel warning
    assert errors.has_warnings()
    assert errors.num_messages() == 1

    # Check that the warning is correct
    warning_info = [*errors.error_info_map.values()][0][0]
    assert warning_info.symbol  == 'x'
    assert warning_info.message == ARRAY_REALLOCATION

    assert f(True) == f2(True)
    assert f(False) == f2(False)

#==============================================================================
def test_stack_array_if(language):

    @stack_array('x')
    def f(b : bool):
        import numpy as np
        if b:
            x = np.array([1,2,3])
        else:
            x = np.array([4,5,6])
        return x[0]

    # Initialize singleton that stores Pyccel errors
    f2 = epyccel(f, language=language)

    assert f(True) == f2(True)
    assert f(False) == f2(False)

#==============================================================================

@pytest.mark.parametrize('lang',[
    pytest.param('fortran', marks = pytest.mark.fortran),
    pytest.param('python', marks = pytest.mark.python),
    pytest.param('c'      , marks = pytest.mark.c)])
def test_Assign_between_nested_If(lang):

    def f(b1 : bool, b2 : bool):
        import numpy as np
        if b1:
            if b2:
                n = 0
            else:
                x = np.zeros(3, dtype=int)
                n = x.shape[0]
        else:
            x = np.zeros(4, dtype=int)
            n = x.shape[0]
        return n

     # Initialize singleton that stores Pyccel errors
    errors = Errors()

    # epyccel should raise an Exception
    f2 = epyccel(f, language=lang)

    # Check that we don't get a Pyccel warning
    assert not errors.has_warnings()

    assert f(True,True) == f2(True,True)
    assert f(True,False) == f2(True,False)
    assert f(False,True) == f2(False,True)

@pytest.mark.skipif(sys.platform == 'win32', reason="NumPy compilation raises warnings on Windows. See issue #1405")
def test_conda_flag_disable(language):
    def one():
        return True
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        epyccel(one, language=language, conda_warnings = 'off')

@pytest.mark.skipif(sys.platform == 'win32', reason="NumPy compilation raises warnings on Windows. See issue #1405")
def test_conda_flag_verbose(language):
    def one():
        return True
    #with pytest.warns(Warning) as record1:
    with warnings.catch_warnings(record=True) as record1:
        warnings.simplefilter("always")
        epyccel(one, language=language, conda_warnings = 'verbose')
    if len(record1)>0:
        warn_message = record1[0].message
        p = str(warn_message).split(":")[2].strip()
        assert p in os.environ['PATH']

#==============================================================================

if __name__ == '__main__':

    for l in ['fortran']:

        test_no_reallocation(l)

        test_reallocation_heap(l)
        test_reallocation_stack(l)

        test_creation_in_loop_heap(l)
        test_creation_in_loop_stack(l)

        test_creation_in_if_heap(l)

    test_Reassign_to_Target()
    test_Assign_Between_Allocatables()
