from .fortrannameclashchecker import FortranNameClashChecker
from .cnameclashchecker import CNameClashChecker
from .pythonnameclashchecker import PythonNameClashChecker

name_clash_checkers = {'fortran':FortranNameClashChecker(),
        'c':CNameClashChecker(),
        'python':PythonNameClashChecker()}
