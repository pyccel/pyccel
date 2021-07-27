import os
import sys

class CompileObj:
    """
    Class containing all information necessary for compiling

    Parameters
    ----------
    file_name     : str
                    Name of file to be compiled

    is_module     : bool
                    Indicates whether we are compiling a module or a program
                    Default : True

    flags         : str
                    Any non-default flags passed to the compiler

    includes      : iterable of strs
                    include directories paths

    libs          : iterable of strs
                    required libraries

    libdirs       : iterable of strs
                    paths to directories containing the required libraries

    dependencies  : iterable of CompileObjs
                    objects which must also be compiled in order to compile this module/program

    accelerators  : str
                    Tool used to accelerate the code (e.g. openmp openacc)
    """
    def __init__(self,
                 file_name,
                 is_module    = True,
                 flags        = (),
                 includes     = (),
                 libs         = (),
                 libdirs      = (),
                 dependencies = (),
                 accelerators = ()):
        if not all(isinstance(d, CompileObj) for d in dependencies):
            raise TypeError("Dependencies require necessary compile information")

        self._file = file_name
        self._module_name = os.path.splitext(file_name)[0]
        if is_module:
            self._target = self._module_name+'.o'
        else:
            self._target = self._module_name
            if sys.platform == "win32":
                self._target += '.o'

        self._flags        = flags or ()
        self._includes     = includes or ()
        self._libs         = libs or ()
        self._libdirs      = libdirs or ()
        self._dependencies = dependencies or ()
        self._accelerators = accelerators or ()
        self._is_module    = is_module

    @property
    def source(self):
        return self._file

    @property
    def module(self):
        return self._module_name

    @property
    def target(self):
        return self._target

    @property
    def flags(self):
        return self._flags

    @property
    def includes(self):
        return self._includes

    @property
    def libs(self):
        return self._libs

    @property
    def libdirs(self):
        return self._libdirs

    @property
    def dependencies(self):
        return self._dependencies

    @property
    def accelerators(self):
        return self._accelerators

    @property
    def is_module(self):
        return self._is_module
