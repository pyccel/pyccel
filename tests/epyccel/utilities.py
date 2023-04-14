# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np
from pytest_teardown_tools import run_epyccel, clean_test


#==============================================================================
class epyccel_test:
    """
    Class which stores a pyccelized function

    This avoids the need to pyccelize the object multiple times
    while still providing a clean interface for the tests
    through the compare_epyccel function
    """
    def __init__(self, f, lang='fortran'):
        self._f  = f
        self._f2 = run_epyccel(f, language=lang)

    def compare_epyccel(self, *args):
        out1 = self._f(*args)
        out2 = self._f2(*args)
        assert np.equal(out1, out2 ).all()
