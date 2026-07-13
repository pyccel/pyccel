# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np

from pyccel import epyccel
from pyccel.errors.errors import PyccelError


# ==============================================================================
class LazyPerFunctionEpyccel:
    """
    Fallback proxy used when a whole-module `epyccel()` translation fails.

    Compiles each attribute individually and caches it, preserving
    per-test failure isolation: a single broken function fails only the
    test(s) that use it, rather than every test sharing the module.
    """

    def __init__(self, pymod, language, epyccel_kwargs):
        self._pymod = pymod
        self._language = language
        self._epyccel_kwargs = epyccel_kwargs
        self._cache = {}

    def __getattr__(self, name):
        if name not in self._cache:
            self._cache[name] = epyccel(
                getattr(self._pymod, name), language=self._language,
                **self._epyccel_kwargs
            )
        return self._cache[name]


# ==============================================================================
def epyccel_module_with_fallback(pymod, language, **kwargs):
    """
    Translate `pymod` as a whole module in one pass.

    If the whole-module translation fails (e.g. `PyccelError` from the
    translator/compiler, or `ImportError` from the shared-library load
    check in `epyccel()`), fall back to lazily translating individual
    functions on first access.
    """
    try:
        return epyccel(pymod, language=language, **kwargs)
    except (PyccelError, ImportError):
        return LazyPerFunctionEpyccel(pymod, language, **kwargs)
