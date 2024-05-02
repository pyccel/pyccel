# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
Module containing all classes which handle name collision rules
for different languages.
"""
from .fortrannameclashchecker import FortranNameClashChecker
from .cnameclashchecker import CNameClashChecker
from .pythonnameclashchecker import PythonNameClashChecker

name_clash_checkers = {'fortran':FortranNameClashChecker(),
        'c':CNameClashChecker(),
        'python':PythonNameClashChecker()}
