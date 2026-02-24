# pylint: disable=missing-function-docstring, missing-module-docstring
import os
import re
import shutil
import subprocess
import sys
from packaging.version import Version

import numpy as np

from pyccel import epyccel
from pyccel.codegen.compiling.compilers import Compiler

__all__ = (
    'epyccel_test',
    'get_compiler_info',
)

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
        self._f2 = epyccel(f, language=lang)

    def compare_epyccel(self, *args):
        out1 = self._f(*args)
        out2 = self._f2(*args)
        assert np.equal(out1, out2 ).all()

#==============================================================================
def get_compiler_info(language):
    """
    Extract the name of the compiler and its version, based on the language.

    Parameters
    ----------
    language : str
        The backend language for Pyccel. Accepted values are 'C', 'Fortran',
        and 'Python' (not case-sensitive).

    Returns
    -------
    executable : str
        The name of the compiler (e.g. 'gcc' or 'gfortran'). If `language` is
        Python, the executable is 'python' by default.

    version : packaging.version.Version
        The compiler version obtained by running `<executable> --version`.
    """
    language = language.lower()
    compiler_family = os.environ.get('PYCCEL_DEFAULT_COMPILER', 'GNU')
    debug = os.environ.get('PYCCEL_DEBUG_MODE', False)

    if language in ['c', 'fortran']:
        compiler = Compiler(compiler_family, debug)
        executable = shutil.which(compiler.compiler_info[language]['exec'])
    else:
        executable = sys.executable

    version_output = subprocess.check_output([executable, '--version']).decode('utf-8')
    version_string = re.search(r"(\d+\.\d+\.\d+)", version_output).group()
    version = Version(version_string)

    return executable, version
