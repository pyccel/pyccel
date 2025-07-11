# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
Module providing objects that are useful for describing the compilation of a project
via the pyccel-make command.
"""

class CompileTarget:
    __slots__ = ('_files', '_is_exe')
    def __init__(self, *files, is_exe):
        self._files = files
        self._is_exe = is_exe
