# pylint: disable=missing-function-docstring
"""
Test the arguments of epyccel itself.
"""
import os
import sys
import warnings
import pytest

from pyccel import epyccel

#---------------------------------------------------------------------
def test_with_capitalised_language(language):
    def free_gift():
        gift = 10
        return gift

    c_gift = epyccel(free_gift, language=language.capitalize())
    assert c_gift() == free_gift()
    assert isinstance(c_gift(), type(free_gift()))

#---------------------------------------------------------------------
def test_with_verbosity(language):
    def free_gift():
        gift = 10
        return gift

    c_gift = epyccel(free_gift, verbose=3)
    assert c_gift() == free_gift()
    assert isinstance(c_gift(), type(free_gift()))

#---------------------------------------------------------------------
@pytest.mark.skipif(sys.platform == 'win32', reason="NumPy compilation raises warnings on Windows. See issue #1405")
def test_conda_flag_disable(language):
    def one():
        return True
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        epyccel(one, language=language, conda_warnings = 'off')

#---------------------------------------------------------------------
@pytest.mark.skipif(sys.platform == 'win32', reason="NumPy compilation raises warnings on Windows. See issue #1405")
def test_conda_flag_verbose(language):
    def one():
        return True
    with warnings.catch_warnings(record=True) as record1:
        warnings.simplefilter("always")
        epyccel(one, language=language, conda_warnings = 'verbose')
    if len(record1)>0:
        warn_message = record1[0].message
        p = str(warn_message).split(":")[2].strip()
        assert p in os.environ['PATH']
