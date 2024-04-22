# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""This file contains different utilities for the Parser."""
import os

from pyccel.ast.variable       import DottedName
from pyccel.ast.internals      import PyccelSymbol

__all__ = ('is_valid_filename_py',
           'is_valid_filename_pyh',
           'get_default_path',
           'pyccel_external_lib')

pyccel_external_lib = {"mpi4py"             : "pyccel.stdlib.external.mpi4py",
                       "scipy.linalg.lapack": "pyccel.stdlib.external.lapack",
                       "scipy.linalg.blas"  : "pyccel.stdlib.external.blas",
                       "scipy.fftpack"      : "pyccel.stdlib.external.dfftpack",
                       "fitpack"            : "pyccel.stdlib.internal.fitpack",
                       "scipy.interpolate._fitpack":"pyccel.stdlib.external.fitpack"}

#==============================================================================

#  ... checking the validity of the filenames, using absolute paths
def _is_valid_filename(filename, ext):
    """Returns True if filename has the extension ext and exists."""

    if not isinstance(filename, str):
        return False

    if not(ext == filename.split('.')[-1]):
        return False
    fname = os.path.abspath(filename)

    return os.path.isfile(fname)

def is_valid_filename_py(filename):
    """Returns True if filename is an existing python file."""
    return _is_valid_filename(filename, 'py')

def is_valid_filename_pyh(filename):
    """Returns True if filename is an existing pyccel header file."""
    return _is_valid_filename(filename, 'pyh')
#  ...

def get_default_path(name):
    """
    Get the full path to the Pyccel description of the imported library.

    This function takes the name of an import source. If the imported library is in
    stdlib, it returns the full Python path to the stdlib equivalent library.
    Otherwise the original name is returned. This equivalent library should be a
    header file which describes all the functions which are supported by Pyccel.

    Parameters
    ----------
    name : PyccelSymbol | DottedName
        The name of the source file for the import.

    Returns
    -------
    PyccelSymbol | DottedName
        The name of the Pyccel-compatible source file for the import.
    """
    name_ = str(name)
    name = pyccel_external_lib.get(name_, name_).split('.')
    if len(name)>1:
        return DottedName(*name)
    else:
        return name[0]


