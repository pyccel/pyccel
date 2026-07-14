# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring

import pytest
from modules import imports
from numpy import ones
from utilities import epyccel_module_with_fallback

from pyccel import epyccel
from pyccel.decorators import inline


@pytest.fixture(scope="module")
def epyc_imports_mod(language):
    return epyccel_module_with_fallback(imports, language)


# ==============================================================================


def test_import(epyc_imports_mod):
    f1 = imports.f1
    f = epyc_imports_mod.f1
    x = ones(10, dtype=int)
    assert f(x) == f1(x)


def test_import_from(epyc_imports_mod):
    f2 = imports.import_from
    f = epyc_imports_mod.import_from
    x = ones(10, dtype=int)
    assert f(x) == f2(x)


def test_import_as(epyc_imports_mod):
    f3 = imports.import_as
    f = epyc_imports_mod.import_as
    x = ones(10, dtype=int)
    assert f(x) == f3(x)


def test_import_method(epyc_imports_mod):
    f5 = imports.import_method
    f = epyc_imports_mod.import_method
    x = ones(10, dtype=int)
    assert f(x) == f5(x)


@pytest.mark.python
def test_import_python_unused_inline():
    import modules.Module_13 as mod

    mod_epyc = epyccel(mod, language="python")
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
        def sin_2(self, d: float):
            return np.sin(2 * d)

        def sin_2_squared(self, d: float):
            return self.sin_2(d) * self.sin_2(d)

    InlineUsingImpEpyc = epyccel(InlineUsingImp, language="python")
    iui = InlineUsingImp()
    val = iui.sin_2(3.0)
    val_squared = iui.sin_2_squared(3.0)
    iui_e = InlineUsingImpEpyc()
    val_e = iui_e.sin_2(3.0)
    val_e_squared = iui_e.sin_2_squared(3.0)
    assert val == val_e
    assert val_squared == val_e_squared
