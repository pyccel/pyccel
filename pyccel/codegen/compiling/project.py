# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
Module providing objects that are useful for describing the compilation of a project
via the pyccel-make command.
"""
from pathlib import Path
from pyccel.errors.errors  import Errors

errors = Errors()

class CompileTarget:
    """
    Class describing a compilation target.

    Class describing the compilation target of a translated Python file.
    The class contains all the information necessary to create the
    necessary targets in a build system (e.g. CMake, meson).

    Parameters
    ----------
    name : str
        The unique identifier for the target.
    pyfile : Path
        The absolute path to the Python file that was translated.
    file : str | Path
        The absolute path to the low-level translation of the Python file.
    wrapper_files : dict[Path, iterable[str]]
        A dictionary whose keys are the absolute paths to the generated wrapper files,
        and whose values are iterables containing the names of the stdlib targets for
        these additional files.
    program_file : str | Path, optional
        The absolute path to the low-level translation of the program found
        in the Python file (if the file contained a program).
    stdlib_deps : iterable[str]
        An iterable containing the names of the stdlib targets of this object.
    """
    __slots__ = ('_name', '_pyfile', '_file', '_wrapper_files',
                 '_program_file', '_dependencies', '_stdlib_deps',
                 '_wrapper_stdlib_deps')
    def __init__(self, name, pyfile, file, wrapper_files, program_file, stdlib_deps):
        self._name = name
        self._pyfile = pyfile
        self._file = Path(file)
        self._wrapper_files = wrapper_files
        self._program_file = Path(program_file) if program_file is not None else program_file
        self._dependencies = []
        self._stdlib_deps = list(stdlib_deps)

    @property
    def name(self):
        """
        The unique identifier for the target.

        The unique identifier for the target.
        """
        return self._name

    @property
    def pyfile(self):
        """
        The absolute path to the Python file that was translated.

        The absolute path to the Python file that was translated.
        """
        return self._pyfile

    @property
    def file(self):
        """
        The absolute path to the low-level translation of the Python file.

        The absolute path to the low-level translation of the Python file.
        """
        return self._file

    @property
    def wrapper_files(self):
        """
        The absolute path to the generated wrapper files.

        The absolute path to the generated wrapper files.
        """
        return self._wrapper_files

    @property
    def program_file(self):
        """
        The absolute path to the low-level translation of the program.

        The absolute path to the low-level translation of the program found
        in the Python file (if the file contained a program).
        """
        return self._program_file

    @property
    def is_exe(self):
        """
        Indicates if an executable should be created from this target.

        Indicates if an executable should be created from this target.
        """
        return self._program_file is not None

    def add_dependencies(self, *new_dependencies):
        """
        Add dependencies to the target.

        Add dependencies to the target. A dependency is something that
        is imported by the file and must therefore be compiled before
        this object.

        Parameters
        ----------
        *new_dependencies : CompileTarget
            The dependencies that should be added.
        """
        self._dependencies.extend(new_dependencies)

    @property
    def dependencies(self):
        """
        Get the dependencies of the target.

        Get all CompileTarget objects describing targets which are imported
        by the file and must therefore be compiled before this object.
        """
        return self._dependencies

    @property
    def stdlib_dependencies(self):
        """
        Get the stdlib dependencies of the target.

        Get a list of strings containing the name of the targets from Pyccel's
        standard library which are required to compile this object.
        """
        return self._stdlib_deps

    def __repr__(self):
        return f'CompileTarget({self.pyfile})'

class DirTarget:
    """
    Class describing a folder containing compilation targets.

    Class describing a folder containing compilation targets. This class sorts
    the compilation targets to ensure they are compiled before they are used.

    Parameters
    ----------
    folder : Path
        The absolute path to the folder containing the generated code.
    compile_targets : iterable[CompileTarget]
        An iterable of the CompileTarget objects which are found in this directory.
    """
    __slots__ = ('_folder', '_targets', '_dependencies')
    def __init__(self, folder, compile_targets : list[CompileTarget]):
        # Group compile targets by subdirectory
        dirs = {}
        for c in compile_targets:
            dir_info = Path(c.pyfile).relative_to(folder).parent.parts
            dirname = dir_info[0] if dir_info else '.'
            dirs.setdefault(folder / dirname, []).append(c)

        for n, c in dirs.items():
            if n == folder:
                continue
            dirs[n] = [DirTarget(n, c)]

        # Find dependencies to calculate the order in which folders should be included
        deps = {}
        for current_folder, compile_objs in dirs.items():
            for f in compile_objs:
                deps[f] = set()
                for c in f.dependencies:
                    if c.pyfile.parent == current_folder:
                        deps[f].add(c.pyfile)
                    elif folder in c.pyfile.parents:
                        deps[f].add(folder / c.pyfile.relative_to(folder).parts[0])

        # Sort folders
        placed = []
        targets = []
        while deps:
            new_target = next((c for (c, d) in deps.items() if all(di in placed for di in d)), None)
            if new_target is None:
                break
            deps.pop(new_target)
            targets.append(new_target)
            if isinstance(new_target, CompileTarget):
                placed.append(new_target.pyfile)
            else:
                placed.append(new_target.folder)

        # If the sorting failed print an error showing the circular dependency
        if deps:
            cycle = [next(c for c in deps)]
            while len(cycle) < 2 or cycle[-1] != cycle[0]:
                c = cycle[-1]
                unfulfilled_dep = next(d for d in deps[c] if d not in placed)
                cycle.append(next(c for c in deps if getattr(c, 'pyfile', getattr(c, 'folder')) == unfulfilled_dep))

            cycle_example = ' -> '.join(str(getattr(c, 'pyfile', c.folder)) for c in cycle)

            errors.report(f"Found circular dependencies between directories: {cycle_example}",
                         severity='fatal')

        self._folder = folder
        self._targets = targets
        self._dependencies = {d for t in self._targets for d in t.dependencies if d not in self}

    @property
    def dependencies(self):
        """
        Get all directories which must be compiled before this directory.

        Get all directories which must be compiled before this directory.
        """
        return self._dependencies

    @property
    def folder(self):
        """
        Get the path to the folder being described by this target.

        Get the path to the folder being described by this target.
        """
        return self._folder

    @property
    def targets(self):
        """
        Get all targets found in this directory.

        Get all targets found in this directory. This includes compilation targets
        and sub-directories.
        """
        return self._targets

    def __contains__(self, other):
        if isinstance(other, CompileTarget):
            return self.folder in other.pyfile.parents
        else:
            return self.folder in other.folder.parents

    def __repr__(self):
        return f'DirTarget({self.folder})'

class BuildProject:
    """
    Class representing the overall build project structure.

    This class encapsulates the directory structure, compilation targets,
    programming languages, and standard library dependencies of a project.
    It serves as the main data container for build configuration.

    Parameters
    ----------
    root_dir : str | Path
        Root directory of the project where the original Python code is found.
    compile_targets : iterable[CompileTarget]
        An iterable of all compile targets in the project.
    languages : iterable[str]
        An iterable of languages used in the project (e.g., ['C', 'Fortran']).
    stdlib_deps : dict[str, CompileObj]
        A dictionary mapping the names of standard library dependencies
        required for the build to the CompileObj describing how they are used.
    """
    def __init__(self, root_dir, compile_targets, languages, stdlib_deps):
        self._root_dir = Path(root_dir)

        self._dir_info = DirTarget(self._root_dir, compile_targets)

        self._languages = languages

        self._stdlib_deps = stdlib_deps

    @property
    def project_name(self):
        """
        Get the name of the project.

        Get the name of the project.
        """
        return self._root_dir.stem

    @property
    def languages(self):
        """
        Get all programming languages used in the project.

        Get all programming languages used in the project.
        """
        return self._languages

    @property
    def stdlib_deps(self):
        """
        Get the dependencies injected by Pyccel.

        Get a dictionary mapping the names of standard library dependencies
        required for the build to the CompileObj describing how they are used.
        """
        return self._stdlib_deps

    @property
    def dir_info(self):
        """
        Get the DirTarget describing the target hierarchy within the project.

        Get the DirTarget describing the target hierarchy within the project.
        """
        return self._dir_info

    @property
    def root_dir(self):
        """
        Get the root directory of the project where the original Python code is found.

        Get the root directory of the project where the original Python code is found.
        """
        return self._root_dir
