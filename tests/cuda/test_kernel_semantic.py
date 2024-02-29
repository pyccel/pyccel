# pylint: disable=missing-function-docstring, missing-module-docstring
import pytest

from pyccel.epyccel import epyccel
from pyccel.decorators import kernel
from pyccel.errors.errors import Errors, PyccelSemanticError
from pyccel.errors.messages import (INVALID_KERNEL_CALL_TP_BLOCK,
                                    INVALID_KERNEL_CALL_BP_GRID,
                                    INVALID_KERNEL_LAUNCH_CONFIG_HEIGHT,
                                    INVALID_KERNEL_LAUNCH_CONFIG_LOW)


@pytest.mark.parametrize('language', [
    pytest.param("cuda", marks=pytest.mark.cuda)
])
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
    assert INVALID_KERNEL_CALL_BP_GRID == error_info.message


@pytest.mark.parametrize('language', [
    pytest.param("cuda", marks=pytest.mark.cuda)
])
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
    assert INVALID_KERNEL_CALL_TP_BLOCK == error_info.message


@pytest.mark.parametrize('language', [
    pytest.param("cuda", marks=pytest.mark.cuda)
])
def test_invalid_launch_config_height(language):
    def invalid_launch_config_height():
        @kernel
        def kernel_call():
            pass

        blocks_per_grid = 1
        threads_per_block = 1
        third_param = 1
        kernel_call[blocks_per_grid, threads_per_block, third_param]()

    errors = Errors()

    with pytest.raises(PyccelSemanticError):
        epyccel(invalid_launch_config_height, language=language)

    assert errors.has_errors()
    assert errors.num_messages() == 1

    error_info = [*errors.error_info_map.values()][0][0]
    assert error_info.symbol.funcdef == 'kernel_call'
    assert INVALID_KERNEL_LAUNCH_CONFIG_HEIGHT == error_info.message


@pytest.mark.parametrize('language', [
    pytest.param("cuda", marks=pytest.mark.cuda)
])
def test_invalid_launch_config_low(language):
    def invalid_launch_config_low():
        @kernel
        def kernel_call():
            pass

        blocks_per_grid = 1
        kernel_call[blocks_per_grid]()

    errors = Errors()

    with pytest.raises(PyccelSemanticError):
        epyccel(invalid_launch_config_low, language=language)

    assert errors.has_errors()
    assert errors.num_messages() == 1

    error_info = [*errors.error_info_map.values()][0][0]
    assert error_info.symbol.funcdef == 'kernel_call'
    assert INVALID_KERNEL_LAUNCH_CONFIG_LOW == error_info.message


@pytest.mark.parametrize('language', [
    pytest.param("cuda", marks=pytest.mark.cuda)
])
def test_invalid_arguments_for_cuda_language(language):
    def invalid_arguments():
        @kernel
        def kernel_call(arg):
            pass

        blocks_per_grid = 1
        threads_per_block = 1
        kernel_call[blocks_per_grid, threads_per_block]()

    errors = Errors()

    with pytest.raises(PyccelSemanticError):
        epyccel(invalid_arguments, language=language)

    assert errors.has_errors()
    assert errors.num_messages() == 1

    error_info = [*errors.error_info_map.values()][0][0]
    assert error_info.symbol.funcdef == 'kernel_call'
    assert "0 argument types given, but function takes 1 arguments" == error_info.message


@pytest.mark.parametrize('language', [
    pytest.param("cuda", marks=pytest.mark.cuda)
])
def test_invalid_arguments_for_cuda_language2(language):
    def invalid_arguments_():
        @kernel
        def kernel_call():
            pass

        blocks_per_grid = 1
        threads_per_block = 1
        kernel_call[blocks_per_grid, threads_per_block](1)

    errors = Errors()

    with pytest.raises(PyccelSemanticError):
        epyccel(invalid_arguments_, language=language)

    assert errors.has_errors()
    assert errors.num_messages() == 1

    error_info = [*errors.error_info_map.values()][0][0]
    assert error_info.symbol.funcdef == 'kernel_call'
    assert "1 argument types given, but function takes 0 arguments" == error_info.message


@pytest.mark.parametrize('language', [
    pytest.param("cuda", marks=pytest.mark.cuda)
])
def test_kernel_return(language):
    def kernel_return():
        @kernel
        def kernel_call():
            return 7

        blocks_per_grid = 1
        threads_per_block = 1
        kernel_call[blocks_per_grid, threads_per_block](1)

    errors = Errors()

    with pytest.raises(PyccelSemanticError):
        epyccel(kernel_return, language=language)

    assert errors.has_errors()
    assert errors.num_messages() == 1

    error_info = [*errors.error_info_map.values()][0][0]
    assert error_info.symbol.funcdef == 'kernel_call'
    assert "cuda kernel function 'kernel_call' returned a value in violation of the laid-down specification" == error_info.message
