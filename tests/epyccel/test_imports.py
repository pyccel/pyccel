# pylint: disable=missing-function-docstring, missing-module-docstring

import pytest
from numpy import ones

from pyccel import epyccel
from pyccel.decorators import inline

#==============================================================================

def test_import(language):
    def f1(x : 'int[:]'):
        import numpy
        s = numpy.shape(x)[0]
        return s

    f = epyccel(f1, language = language)
    x = ones(10, dtype=int)
    assert f(x) == f1(x)

def test_import_from(language):
    def f2(x : 'int[:]'):
        from numpy import shape
        s = shape(x)[0]
        return s


    f = epyccel(f2, language = language)
    x = ones(10, dtype=int)
    assert f(x) == f2(x)

def test_import_as(language):
    def f3(x : 'int[:]'):
        import numpy as np
        s = np.shape(x)[0]
        return s

    f = epyccel(f3, language = language)
    x = ones(10, dtype=int)
    assert f(x) == f3(x)

def test_import_method(language):
    def f5(x : 'int[:]'):
        s = x.shape[0]
        return s

    f = epyccel(f5, language = language)
    x = ones(10, dtype=int)
    assert f(x) == f5(x)

@pytest.mark.python
def test_import_python_unused_inline():
    import modules.Module_13 as mod
    mod_epyc = epyccel(mod, language='python')
    ui = mod.UnusedInline()
    val = ui.sin_2(3.0)
    ui_e = mod_epyc.UnusedInline()
    val_e = ui_e.sin_2(3.0)
    assert val == val_e

@pytest.mark.python
def test_import_python_inline():
    import numpy as np

    class InlineUsingImp:
        @inline
        def sin_2(self, d : float):
            return np.sin(2 * d)

        def sin_2_squared(self, d : float):
            return self.sin_2(d) * self.sin_2(d)

    InlineUsingImpEpyc = epyccel(InlineUsingImp, language='python')
    iui = InlineUsingImp()
    val = iui.sin_2(3.0)
    val_squared = iui.sin_2_squared(3.0)
    iui_e = InlineUsingImpEpyc()
    val_e = iui_e.sin_2(3.0)
    val_e_squared = iui_e.sin_2_squared(3.0)
    assert val == val_e
    assert val_squared == val_e_squared
