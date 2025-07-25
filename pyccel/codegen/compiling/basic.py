#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
Module handling classes for compiler information relevant to a given object
"""
from pathlib import Path
import sys
from filelock import FileLock

class CompileObj:
    """
    Class containing all information necessary for compiling.

    A class which stores all information which may be needed in order to
    compile an object. This includes its name, location, and all dependencies
    and flags which may need to be passed to the compiler.

    Parameters
    ----------
    file_name : str
        Name of file to be compiled.

    dirpath : str
        Path to the directory where the file is found.

    flags : str
        Any non-default flags passed to the compiler.

    include : iterable of strs
        Include directories paths.

    libs : iterable of strs
        Required libraries.

    libdir : iterable of strs
        Paths to directories containing the required libraries.

    dependencies : iterable of CompileObjs
        Objects which must also be compiled in order to compile this module/program.

    extra_compilation_tools : iterable of str
        Tools used which require additional compilation flags/include dirs/libs/etc.

    has_target_file : bool, default : True
        If set to false then this flag indicates that the file has no target.
        Eg an interface for a library.

    prog_target : str, default: None
        The name of the executable that should be generated if this file is a
        program. If no name is provided then the module name deduced from the file
        name is used.
    """
    compilation_in_progress = FileLock('.lock_acquisition.lock')
    __slots__ = ('_file_path','_dirpath','_module_name','_module_target','_prog_target',
                 '_lock_target','_lock_source','_flags','_include','_libs',
                 '_libdir','_extra_compilation_tools','_dependencies','_has_target_file')
    def __init__(self,
                 file_name,
                 dirpath,
                 flags        = (),
                 include     = (),
                 libs         = (),
                 libdir      = (),
                 dependencies = (),
                 extra_compilation_tools = (),
                 has_target_file = True,
                 prog_target  = None):

        dirpath = Path(dirpath)
        self._dirpath = dirpath
        self._file_path = dirpath / file_name

        self._module_name = Path(file_name).stem
        rel_mod_name = dirpath / self._module_name
        self._module_target = rel_mod_name.with_suffix('.o')

        if prog_target:
            self._prog_target = prog_target
        else:
            self._prog_target = self._module_name
        if sys.platform == "win32":
            self._prog_target = self._prog_target + '.exe'

        self._lock_target  = FileLock(str(self.module_target.with_suffix(
                                            self.module_target.suffix + '.lock')))
        self._lock_source  = FileLock(str(self.source.with_suffix(
                                            self.source.suffix + '.lock')))

        self._flags        = list(flags)
        self._include     = {dirpath, *(Path(i) for i in include)}
        self._libs         = list(libs)
        self._libdir      = set(libdir)
        self._extra_compilation_tools = set(extra_compilation_tools)
        self._dependencies = {a.module_target:a for a in dependencies}
        self._has_target_file = has_target_file

    def reset_dirpath(self, dirpath):
        """
        Change the dirpath in which the source file is saved.

        Change the dirpath in which the source file is saved. Normally the location
        of the source file should not change during the execution, however when
        working with the stdlib, the `CompileObj` is created with the dirpath set
        to the file's location in the Pyccel install directory. When the file is
        used it is copied to the user's dirpath, at which point the dirpath of the
        `CompileObj` must be updated.

        Parameters
        ----------
        dirpath : str
            The new dirpath where the source file can be found.
        """
        dirpath = Path(dirpath)
        self._include.remove(self._dirpath)
        self._include.add(dirpath)

        self._file_path = dirpath / self._file_path.name
        self._lock_source  = FileLock(self.source.with_suffix(
                                        self.source.suffix+'.lock'))
        self._dirpath = dirpath
        self._include.add(self._dirpath)

        rel_mod_name = dirpath / self._module_name
        self._module_target = rel_mod_name.with_suffix('.o')

        self._prog_target = rel_mod_name
        if sys.platform == "win32":
            self._prog_target.with_suffix('.exe')

        self._lock_target = FileLock(self.module_target.with_suffix(
                                        self.module_target.suffix+'.lock'))

    @property
    def source(self):
        """ Returns the file to be compiled
        """
        return self._file_path

    @property
    def source_dirpath(self):
        """ Returns the location of the file to be compiled
        """
        return self._dirpath

    @property
    def python_module(self):
        """ Returns the python name of the file to be compiled
        """
        return self._module_name

    @property
    def module_target(self):
        """ Returns the .o file to be generated by the compilation step
        """
        return self._module_target

    @property
    def program_target(self):
        """ Returns the program to be generated by the compilation step
        """
        return self._prog_target

    @property
    def flags(self):
        """ Returns the additional flags required to compile the file
        """
        return self._flags

    @property
    def include(self):
        """
        Get the additional include directories required to compile the file.

        Return a set containing all the directories which must be passed to the
        compiler via the include flag `-I`.
        """
        return self._include.union([di for d in self._dependencies.values() for di in d.include])

    @property
    def libs(self):
        """
        Get the additional libraries required to compile the file.

        Return a list containing all the libraries which must be passed to the
        compiler via the library flag `-l`.
        """
        return self._libs+[dl for d in self._dependencies.values() for dl in d.libs]

    @property
    def libdir(self):
        """
        Get the additional library directories required to compile the file.

        Return a set containing all the directories which must be passed to the
        compiler via the library directory flag `-L` so that the necessary
        libraries can be correctly located.
        """
        return self._libdir.union([dld for d in self._dependencies.values() for dld in d.libdir])

    @property
    def extra_modules(self):
        """ Returns the additional objects required to compile the file
        """
        deps = set()
        for d in self._dependencies.values():
            if d.has_target_file:
                deps.add(d.module_target)
                deps.update(d.extra_modules)
        return deps

    @property
    def dependencies(self):
        """ Returns the objects which the file to be compiled uses
        """
        return self._dependencies.values()

    def get_dependency(self, target):
        """ Returns the objects which the file to be compiled uses
        """
        return self._dependencies.get(target, None)

    def add_dependencies(self, *args):
        """
        Indicate that the file to be compiled depends on a given other file

        Parameters
        ----------
        *args : CompileObj
        """
        if not all(isinstance(d, CompileObj) for d in args):
            raise TypeError("Dependencies require necessary compile information")
        self._dependencies.update({a.module_target:a for a in args})

    def __enter__(self):
        self.compilation_in_progress.acquire()
        self.acquire_lock()

    def acquire_lock(self):
        """
        Lock the file and its dependencies to prevent race conditions.

        Acquire the file locks for the file being compiled, all dependencies needed
        to compile it and the target file which will be generated.
        """
        self._lock_source.acquire()
        self.acquire_simple_lock()
        for d in self.dependencies:
            d.acquire_simple_lock()

    def acquire_simple_lock(self):
        """
        Lock the file created by this `CompileObj`.

        Acquire the file lock for the file created by this `CompileObj` to prevent
        race conditions. This function should be called when the created file is a
        dependency, it is therefore not necessary for it to recurse into its own
        dependencies.
        """
        if self.has_target_file:
            self._lock_target.acquire()

    def __exit__(self, exc_type, value, traceback):
        self.release_lock()
        self.compilation_in_progress.release()

    def release_lock(self):
        """
        Unlock the file and its dependencies.

        Release the file locks for the file being compiled, all dependencies needed
        to compile it and the target file which will be generated.
        """
        for d in self.dependencies:
            d.release_simple_lock()
        self._lock_source.release()
        self.release_simple_lock()

    def release_simple_lock(self):
        """
        Unlock the file created by this `CompileObj`.

        Release the file lock for the file created by this `CompileObj` to prevent
        race conditions. This function should be called when the created file is a
        dependency, it is therefore not necessary for it to recurse into its own
        dependencies.
        """
        if self.has_target_file:
            self._lock_target.release()

    @property
    def extra_compilation_tools(self):
        """
        The name of tools used which require additional compilation information.

        Return a set containing the name of all tools required additional
        information to compile the file. This additional informationcan take the
        form of flags, include directories, libraries, orr library directories.
        Examples of 'extra_compilation_tools' are: openmp, openacc, python.
        """
        return self._extra_compilation_tools.union([da for d in self._dependencies.values() \
                                                       for da in d.extra_compilation_tools])

    def __eq__(self, other):
        return self.module_target == other.module_target

    def __hash__(self):
        return hash(self.module_target)

    @property
    def has_target_file(self):
        """
        Indicates whether the file has a target.
        Eg an interface for a library may not have a target
        """
        return self._has_target_file
