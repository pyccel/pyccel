# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

from .ccode import CCodePrinter, import_dict
from pyccel.ast.variable import DottedName

__all__ = ["CCudaCodePrinter", "cudaccode"]

cuda_library_headers = {
    "cuda_runtime",
    "stdint",
    # ...
}

class CCudaCodePrinter(CCodePrinter):
    """ Cuda C Code printer
    """


def cudaccode():
    pass
