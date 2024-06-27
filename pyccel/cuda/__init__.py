#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
    This module is for exposing the CudaSubmodule functions.
"""
from .cuda_sync_primitives    import synchronize

__all__ = ['synchronize']
