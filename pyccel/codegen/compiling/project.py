# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
Module providing objects that are useful for describing the compilation of a project
via the pyccel-make command.
"""
import pathlib

class CompileTarget:
    #__slots__ = ('_name', '_file', '_wrapper_files', '_program_file', '_targets')
    def __init__(self, name, pyfile, file, wrapper_files, program_file, stdlib_deps):
        self._name = name
        self._pyfile = pyfile
        self._file = pathlib.Path(file)
        self._wrapper_files = wrapper_files
        self._program_file = pathlib.Path(program_file) if program_file is not None else program_file
        self._dependencies = []
        self._stdlib_deps = list(stdlib_deps)

    @property
    def name(self):
        return self._name

    @property
    def pyfile(self):
        return self._pyfile

    @property
    def file(self):
        return self._file

    @property
    def wrapper_files(self):
        return self._wrapper_files

    @property
    def program_file(self):
        return self._program_file

    @property
    def is_exe(self):
        return self._program_file is not None

    def add_dependencies(self, dependencies_iterable):
        self._dependencies.extend(dependencies_iterable)

    @property
    def dependencies(self):
        return self._dependencies

    @property
    def stdlib_dependencies(self):
        return self._dependencies

    def __repr__(self):
        return f'CompileTarget({self.name})'

    @property
    def path_parts(self):
        return self.pyfile.parts

    @property
    def path(self):
        return self.pyfile

class DirTarget:

    def __init__(self, folder, compile_targets : list[CompileTarget]):
        dirs = {}
        for c in compile_targets:
            dir_info = pathlib.Path(c.pyfile).relative_to(folder).parent.parts
            dirname = dir_info[0] if dir_info else '.'
            dirs.setdefault(folder / dirname, []).append(c)

        deps = {}
        if folder in dirs:
            for f in dirs[folder]:
                deps[f] = []
                for c in f.dependencies:
                    if c.pyfile.parent == folder:
                        deps[f].append(c.pyfile)
                    elif folder in c.pyfile.parents:
                        deps[f].append(folder / c.pyfile.relative_to(folder).parts[0])

        for n, c in dirs.items():
            if n == folder:
                continue
            dir_target = DirTarget(n, c)
            deps[dir_target] = []
            for d in dir_target.dependencies:
                assert isinstance(d, CompileTarget)
                if d.pyfile.parent == folder:
                    deps[dir_target].append(d.pyfile)
                elif folder in d.pyfile.parents:
                    deps[dir_target].append(folder / d.pyfile.relative_to(folder).parts[0])

        placed = []
        targets = []
        while deps:
            new_target = next(c for (c, d) in deps.items() if all(di in placed for di in d))
            deps.pop(new_target)
            targets.append(new_target)
            placed.append(new_target.path)

        self._folder = folder
        self._targets = targets
        self._dependencies = {d for t in self._targets for d in t.dependencies if d not in self}

    @property
    def dependencies(self):
        return self._dependencies

    @property
    def folder(self):
        return self._folder

    @property
    def targets(self):
        return self._targets

    def __contains__(self, other):
        if isinstance(other, CompileTarget):
            return self.folder in other.pyfile.parents
        else:
            return self.folder in other.folder.parents

    @property
    def path_parts(self):
        return self.folder.parts

    @property
    def path(self):
        return self.folder

class BuildProject:
    def __init__(self, root_dir, compile_targets, languages, stdlib_deps):
        self._root_dir = pathlib.Path(root_dir)

        self._dir_info = DirTarget(self._root_dir, compile_targets)

        self._languages = languages

        self._stdlib_deps = stdlib_deps

    @property
    def project_name(self):
        return self._root_dir.stem

    @property
    def languages(self):
        return self._languages

    @property
    def stdlib_deps(self):
        return self._stdlib_deps

    @property
    def dir_info(self):
        return self._dir_info

    @property
    def root_dir(self):
        return self._root_dir
