import os
import sys
from filelock import FileLock

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
                 folder,
                 is_module    = True,
                 flags        = (),
                 includes     = (),
                 libs         = (),
                 libdirs      = (),
                 dependencies = (),
                 accelerators = ()):
        if not all(isinstance(d, CompileObj) for d in dependencies):
            raise TypeError("Dependencies require necessary compile information")

        self._file = os.path.join(folder, file_name)
        self._folder = folder
        self._module_name = os.path.splitext(file_name)[0]
        if is_module:
            self._target = self._module_name+'.o'
        else:
            self._target = self._module_name
            if sys.platform == "win32":
                self._target += '.o'
        self._target = os.path.join(folder, self._target)

        self._flags        = list(flags or ())
        self._includes     = [folder, *(includes or ())]
        self._libs         = list(libs or ())
        self._libdirs      = list(libdirs or ())
        self._accelerators = set(accelerators or ())
        self._dependencies = []
        if dependencies:
            self.add_dependencies(*dependencies)
        self._is_module    = is_module
        self._lock         = FileLock(self.python_module+'.lock')

    @property
    def source(self):
        return self._file

    @property
    def source_folder(self):
        return self._folder

    @property
    def python_module(self):
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
    def extra_modules(self):
        deps = set(d.target for d in self._dependencies)
        for d in self._dependencies:
            deps = deps.union(d.extra_modules)
        return deps

    @property
    def dependencies(self):
        return self._dependencies

    def add_dependencies(self, *args):
        if not all(isinstance(d, CompileObj) for d in args):
            raise TypeError("Dependencies require necessary compile information")
        self._dependencies.extend(args)
        for a in args:
            self._flags.extend(a.flags)
            self._includes.extend(a.includes)
            self._libs.extend(a.libs)
            self._libdirs.extend(a.libdirs)
            self._accelerators.union(a.accelerators)

    def acquire_lock(self):
        self._lock.acquire()
        for d in self.dependencies:
            d.acquire_lock()

    def release_lock(self):
        self._lock.release()
        for d in self.dependencies:
            d.release_lock()

    @property
    def accelerators(self):
        return self._accelerators

    @property
    def is_module(self):
        return self._is_module
