#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
Module for introspecting information on Pyccel, in the codebase and the tests.
"""
import os
import re
import subprocess
import sys

from packaging.version import Version

from pyccel.codegen.compiling.compilers import Compiler

__all__ = (
    'get_compiler_info',
)

#==============================================================================
def get_compiler_info(language):
    """
    Extract the path to the compiler and its version, based on the language.

    Extract the path to the compiler and its version, based on the language.

    Parameters
    ----------
    language : str
        The backend language for Pyccel. Accepted values are 'C', 'Fortran',
        and 'Python' (not case-sensitive).

    Returns
    -------
    executable : str
        The path to the compiler (e.g. 'gcc' or 'gfortran'). If `language` is
        Python, the executable is the current Python executable.

    version : packaging.version.Version
        The compiler version obtained by running `<executable> --version`.
    """
    language = language.lower()
    compiler_family = os.environ.get('PYCCEL_DEFAULT_COMPILER', 'GNU')
    debug = os.environ.get('PYCCEL_DEBUG_MODE', False)

    if language == 'python':
        executable = sys.executable
    else:
        compiler = Compiler(compiler_family, debug)
        try:
            executable = compiler.get_exec((), language)
        except KeyError:
            raise ValueError(f"language '{language}' not supported for compiler {compiler_family}") #pylint: disable=raise-missing-from

    version_output = subprocess.check_output([executable, '--version']).decode('utf-8')
    version_string = re.search(r"(\d+\.\d+\.\d+)", version_output).group()
    version = Version(version_string)

    return executable, version
