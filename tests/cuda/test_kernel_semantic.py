
# pylint: disable=missing-function-docstring, missing-module-docstring/
import pytest

from pyccel.epyccel import epyccel
from pyccel.decorators import  kernel
from pyccel.errors.errors import Errors, PyccelSemanticError
from pyccel.errors.messages import (INVALID_KERNEL_CALL_TP_BLOCK,
                                    INVALID_KERNEL_CALL_BP_GRID,
                                    INVALID_KERNEL_LAUNCH_CONFIG,
                                    INVALID_FUNCTION_CALL,
                                    )
@pytest.mark.parametrize( 'language', [
        pytest.param("cuda", marks = pytest.mark.cuda)
    ]
)
def test_invalid_block_number(language):
    def invalid_block_number():
        @kernel
        def kernel_call():
            pass
        blocks_per_grid = 1.0
        threads_per_block = 1
        kernel_call[blocks_per_grid, threads_per_block]()

    errors = Errors()

    with pytest.raises(PyccelSemanticError):
        epyccel(invalid_block_number, language=language)

    assert errors.has_errors()
    assert errors.num_messages() == 1

    error_info = [*errors.error_info_map.values()][0][0]
    assert error_info.symbol.funcdef == 'kernel_call'
    assert  INVALID_KERNEL_CALL_BP_GRID == error_info.message

@pytest.mark.parametrize( 'language', [
        pytest.param("cuda", marks = pytest.mark.cuda)
    ]
)
def test_invalid_thread_per_block(language):
    def invalid_thread_per_block():
        @kernel
        def kernel_call():
            pass
        blocks_per_grid = 1
        threads_per_block = 1.0
        kernel_call[blocks_per_grid, threads_per_block]()

    errors = Errors()

    with pytest.raises(PyccelSemanticError):
        epyccel(invalid_thread_per_block, language=language)

    assert errors.has_errors()
    assert errors.num_messages() == 1

    error_info = [*errors.error_info_map.values()][0][0]
    assert error_info.symbol.funcdef == 'kernel_call'
    assert  INVALID_KERNEL_CALL_TP_BLOCK == error_info.message

@pytest.mark.parametrize( 'language', [
        pytest.param("cuda", marks = pytest.mark.cuda)
    ]
)
def test_invalid_launch_config(language)
    def invalid_launch_config():
        @kernel
        def kernel_call():
            pass
        blocks_per_grid = 1
        threads_per_block = 1
        third_param = 1
        kernel_call[blocks_per_grid, threads_per_block, third_param]

    errors = Errors()

    with pytest.raises(PyccelSemanticError):
        epyccel(invalid_thread_per_block, language=language)

    assert errors.has_errors()
    assert errors.num_messages() == 1

    error_info = [*errors.error_info_map.values()][0][0]
    assert error_info.symbol.funcdef == 'kernel_call'
    assert  INVALID_KERNEL_LAUNCH_CONFIG == error_info.message
    
@pytest.mark.parametrize( 'language', [
        pytest.param("cuda", marks = pytest.mark.cuda)
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
