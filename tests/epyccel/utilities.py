# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np

from pyccel import epyccel


# ==============================================================================
class epyccel_test:
    """
    Class which stores a pyccelized function

    This avoids the need to pyccelize the object multiple times
    while still providing a clean interface for the tests
    through the compare_epyccel function
    """

    def __init__(self, f, lang="fortran"):
        self._f = f
        self._f2 = epyccel(f, language=lang)

    def compare_epyccel(self, *args):
        out1 = self._f(*args)
        out2 = self._f2(*args)
        assert np.equal(out1, out2).all()


# ==============================================================================
class _LazyPerFunctionEpyccel:
    """
    Fallback proxy used when a whole-module `epyccel()` translation fails.

    Compiles each attribute individually and caches it, preserving
    per-test failure isolation: a single broken function fails only the
    test(s) that use it, rather than every test sharing the module.
    """

    def __init__(self, pymod, language):
        self._pymod = pymod
        self._language = language
        self._cache = {}

    def __getattr__(self, name):
        if name not in self._cache:
            print("Translating function")
            self._cache[name] = epyccel(
                getattr(self._pymod, name), language=self._language
            )
        return self._cache[name]


# ==============================================================================
def epyccel_module_with_fallback(pymod, language):
    """
    Translate `pymod` as a whole module in one pass.

    If the whole-module translation fails (e.g. `PyccelError` from the
    translator/compiler, or `ImportError` from the shared-library load
    check in `epyccel()`), fall back to lazily translating individual
    functions on first access.
    """
    try:
        return epyccel(pymod, language=language)
    except Exception:
        return _LazyPerFunctionEpyccel(pymod, language)
