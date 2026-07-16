# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np

from pyccel import epyccel
from pyccel.errors.errors import PyccelError


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
class LazyPerFunctionEpyccel:
    """
    Fallback proxy used when a whole-module `epyccel()` translation fails.

    Compiles each attribute individually and caches it, preserving
    per-test failure isolation: a single broken function fails only the
    test(s) that use it, rather than every test sharing the module.
    """

    def __init__(self, pymod, language, epyccel_kwargs=None):
        self._pymod = pymod
        self._language = language
        self._epyccel_kwargs = epyccel_kwargs or {}
        self._cache = {}

    def __getattr__(self, name):
        if name not in self._cache:
            self._cache[name] = epyccel(
                getattr(self._pymod, name),
                language=self._language,
                **self._epyccel_kwargs,
            )
        return self._cache[name]

    @property
    def language(self):
        """
        str: The language we are translating to.
        """
        return self._language


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
        mod = epyccel(pymod, language=language, **kwargs)
    except (PyccelError, ImportError):
        return LazyPerFunctionEpyccel(pymod, language, kwargs)
    mod.language = language
    return mod


# ==============================================================================
def compare_epyccel(f, language, *args):
    """
    Pyccelize `f`, call both versions with `args`, and assert the outputs match.
    """
    f2 = epyccel(f, language=language)
    out1 = f(*args)
    out2 = f2(*args)
    if isinstance(out1, tuple):
        assert all(np.array_equal(r1, r2).all() for r1, r2 in zip(out1, out2))
    else:
        assert np.array_equal(out1, out2).all()


# ==============================================================================
def matching_types(pyccel_result, python_result):
    """Returns True if the types match, False otherwise"""
    if type(pyccel_result) is type(python_result):
        return True
    return (
        isinstance(pyccel_result, bool) and isinstance(python_result, np.bool_)
    ) or (isinstance(pyccel_result, np.int32) and isinstance(python_result, np.intc))
