# pylint: disable=missing-function-docstring, missing-module-docstring
import pytest

from pyccel.epyccel import epyccel
from pyccel.decorators import kernel, device
from pyccel.errors.errors import Errors, PyccelSemanticError
from pyccel.errors.messages import (MISSING_KERNEL_CONFIGURATION,
                                    )

@pytest.mark.cuda
def test_invalid_launch_configuration():
    def invalid_launch_configuration():
        @device
        def add_one(x : int)
            return x + 1

        x = add_one(1)

    errors = Errors()

    with pytest.raises(PyccelSemanticError):
        epyccel(invalid_block_number, language="cuda")

    assert errors.has_errors()
    assert errors.num_messages() == 1

    error_info = [*errors.error_info_map.values()][0][0]
    assert MISSING_KERNEL_CONFIGURATION == error_info.message
