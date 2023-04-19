
# pylint: disable=missing-function-docstring, missing-module-docstring/
import pytest

import numpy as np
from pyccel.epyccel import epyccel
from pyccel.decorators import stack_array, types, kernel
from pyccel.errors.errors import Errors, PyccelSemanticError
from pyccel.errors.messages import (INVALID_FUNCTION_CALL, KERNEL_STACK_ARRAY_ARG, MISSING_KERNEL_CONFIGURATION,
                                    NON_KERNEL_FUNCTION_CUDA_VAR,
                                    INVALID_KERNEL_CALL_BP_GRID,
                                    INVALID_KERNEL_CALL_TP_BLOCK
                                    )

@pytest.mark.parametrize( 'language', [
        pytest.param("ccuda", marks = pytest.mark.ccuda)
    ]
)
def test_stack_array_kernel(language):
    @stack_array('arr')
    def kernel_caller():
        from numpy import ones
        @kernel
        @types('int[:]')
        def stack_array_kernel(arr):
            return arr[0]
        arr = ones(1, dtype=int)
        return stack_array_kernel[1,1](arr)

    # Initialize singleton that stores Pyccel errors
    errors = Errors()

    # epyccel should raise an Exception
    with pytest.raises(PyccelSemanticError):
        epyccel(kernel_caller, language=language)

    # Check that we got exactly 1 Pyccel error
    assert errors.has_errors()
    assert errors.num_messages() == 1

    # Check that the error is correct
    error_info = [*errors.error_info_map.values()][0][0]
    assert error_info.symbol.funcdef  == 'stack_array_kernel'
    assert KERNEL_STACK_ARRAY_ARG == error_info.message

@pytest.mark.parametrize( 'language', [
        pytest.param("ccuda", marks = pytest.mark.ccuda)
    ]
)
def test_cuda_intern_var_non_kernel(language):
    def non_kernel_function():
        from pyccel import cuda
        i = cuda.threadIdx(0) + cuda.blockIdx(0) * cuda.blockDim(0)

    # Initialize singleton that stores Pyccel errors
    errors = Errors()

    # epyccel should raise an Exception
    with pytest.raises(PyccelSemanticError):
        epyccel(non_kernel_function, language=language)

    # Check that we got exactly 1 Pyccel error
    assert errors.has_errors()
    assert errors.num_messages() == 1

    # Check that the error is correct
    error_info = [*errors.error_info_map.values()][0][0]
    assert error_info.symbol.name[0]            == 'cuda'
    assert error_info.symbol.name[1].func_name  == 'threadIdx'
    assert NON_KERNEL_FUNCTION_CUDA_VAR == error_info.message

@pytest.mark.parametrize( 'language', [
        pytest.param("ccuda", marks = pytest.mark.ccuda)
    ]
)
def test_unvalid_block_number(language):
    def unvalid_block_number():
        @kernel
        def kernel_call():
            pass
        kernel_call[1.2,1]()

    # Initialize singleton that stores Pyccel errors
    errors = Errors()

    # epyccel should raise an Exception
    with pytest.raises(PyccelSemanticError):
        epyccel(unvalid_block_number, language=language)

    # Check that we got exactly 1 Pyccel error
    assert errors.has_errors()
    assert errors.num_messages() == 1

    # Check that the error is correct
    error_info = [*errors.error_info_map.values()][0][0]
    assert error_info.symbol.funcdef  == 'kernel_call'
    assert INVALID_KERNEL_CALL_BP_GRID == error_info.message

@pytest.mark.parametrize( 'language', [
        pytest.param("ccuda", marks = pytest.mark.ccuda)
    ]
)
def test_unvalid_thread_per_block(language):
    def unvalid_thread_per_block():
        @kernel
        def kernel_call():
            pass
        kernel_call[1,1.2]()

    # Initialize singleton that stores Pyccel errors
    errors = Errors()

    # epyccel should raise an Exception
    with pytest.raises(PyccelSemanticError):
        epyccel(unvalid_thread_per_block, language=language)

    # Check that we got exactly 1 Pyccel error
    assert errors.has_errors()
    assert errors.num_messages() == 1

    # Check that the error is correct
    error_info = [*errors.error_info_map.values()][0][0]
    assert error_info.symbol.funcdef  == 'kernel_call'
    assert INVALID_KERNEL_CALL_TP_BLOCK == error_info.message

@pytest.mark.parametrize( 'language', [
        pytest.param("ccuda", marks = pytest.mark.ccuda)
    ]
)
def test_missing_kernel_config(language):
    def missing_kernel_config():
        @kernel
        def kernel_call():
            pass
        kernel_call()

    errors = Errors()

    with pytest.raises(PyccelSemanticError):
        epyccel(missing_kernel_config, language=language)

    assert errors.has_errors()
    assert errors.num_messages() == 1

    error_info = [*errors.error_info_map.values()][0][0]
    assert error_info.symbol.func_name == 'kernel_call'
    assert MISSING_KERNEL_CONFIGURATION == error_info.message

@pytest.mark.parametrize( 'language', [
        pytest.param("ccuda", marks = pytest.mark.ccuda)
    ]
)
def test_invalid_block_number(language):
    def invalid_block_number():
        @kernel
        def kernel_call():
            pass
        blocks_per_grid = 5.0
        kernel_call[blocks_per_grid, 1]()

    errors = Errors()

    with pytest.raises(PyccelSemanticError):
        epyccel(invalid_block_number, language=language)

    assert errors.has_errors()
    assert errors.num_messages() == 1

    error_info = [*errors.error_info_map.values()][0][0]
    assert error_info.symbol.funcdef == 'kernel_call'
    assert  INVALID_KERNEL_CALL_BP_GRID == error_info.message

@pytest.mark.parametrize( 'language', [
        pytest.param("ccuda", marks = pytest.mark.ccuda)
    ]
)
def test_invalid_thread_per_block(language):
    def invalid_thread_per_block():
        @kernel
        def kernel_call():
            pass
        threads_per_block = 5.0
        kernel_call[1, threads_per_block]()

    errors = Errors()

    with pytest.raises(PyccelSemanticError):
        epyccel(invalid_thread_per_block, language=language)

    assert errors.has_errors()
    assert errors.num_messages() == 1

    error_info = [*errors.error_info_map.values()][0][0]
    assert error_info.symbol.funcdef == 'kernel_call'
    assert  INVALID_KERNEL_CALL_TP_BLOCK == error_info.message


@pytest.mark.parametrize( 'language', [
        pytest.param("ccuda", marks = pytest.mark.ccuda)
    ]
)
def test_invalid_function_call(language):
    def invalid_function_call():
        def non_kernel_func():
            pass
        non_kernel_func[1, 2]() # pylint: disable=E1136

    errors = Errors()

    with pytest.raises(PyccelSemanticError):
        epyccel(invalid_function_call, language=language)

    assert errors.has_errors()
    assert errors.num_messages() == 1

    error_info = [*errors.error_info_map.values()][0][0]
    assert error_info.symbol.funcdef == 'non_kernel_func'
    assert INVALID_FUNCTION_CALL == error_info.message
