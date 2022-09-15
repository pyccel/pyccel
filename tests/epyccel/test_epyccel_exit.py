# pylint: disable=missing-function-docstring, missing-module-docstring/
from utilities import getExitStatus
from pyccel.decorators import types
from pyccel.epyccel import epyccel

def test_empty_exit(language):
    def f_call():
        import sys
        sys.exit()

    f = epyccel(f_call, language=language)
    assert getExitStatus(f_call) == getExitStatus(f)


def test_negative_exit(language):
    @types('int')
    def f_call(code):
        import sys
        sys.exit(code)

    f = epyccel(f_call, language=language)
    assert getExitStatus(f_call, -1) == getExitStatus(f, -1)
    assert getExitStatus(f_call, -5) == getExitStatus(f, -5)

def test_zero_exit(language):
    @types('int')
    def f_call(code):
        import sys
        sys.exit(code)

    def f_call_literals():
        import sys
        sys.exit(0)

    f = epyccel(f_call, language=language)
    f_literals = epyccel(f_call_literals, language=language)
    assert getExitStatus(f_call, 0) == getExitStatus(f, 0)
    assert getExitStatus(f_call_literals) == getExitStatus(f_literals)


def test_positive_exit(language):
    @types('int')
    def f_call(code):
        import sys
        sys.exit(code)

    f = epyccel(f_call, language=language)
    assert getExitStatus(f_call, 1) == getExitStatus(f, 1)
    assert getExitStatus(f, 5) == 5
    assert getExitStatus(f_call, 1024) == getExitStatus(f, 1024)
    assert getExitStatus(f_call, 2147483647) == getExitStatus(f, 2147483647)
