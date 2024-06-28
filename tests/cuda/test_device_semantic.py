# pylint: disable=missing-function-docstring, missing-module-docstring
import pytest

from pyccel import epyccel
from pyccel.decorators import device
from pyccel.errors.errors import Errors, PyccelSemanticError
from pyccel.errors.messages import (INVAlID_DEVICE_CALL,)


@pytest.mark.cuda
def test_invalid_device_call():
    def invalid_device_call():
        @device
        def device_call():
            pass
        def fake_kernel_call():
            device_call()

        fake_kernel_call()

    errors = Errors()

    with pytest.raises(PyccelSemanticError):
        epyccel(invalid_device_call, language="cuda")

    assert errors.has_errors()

    assert errors.num_messages() == 1

    error_info = [*errors.error_info_map.values()][0][0]
    assert INVAlID_DEVICE_CALL == error_info.message
