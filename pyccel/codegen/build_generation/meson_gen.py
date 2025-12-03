"""
A module to handle everything related to meson for the pyccel-make command.
"""
from itertools import chain
import os
from pathlib import Path
import shutil
import subprocess
import sys

from pyccel.codegen.compiling.project import DirTarget
from pyccel.codegen.compiling.library_config import recognised_libs, ExternalLibInstaller

from .build_gen import BuildSystemHandler

class MesonHandler(BuildSystemHandler):
    """
    A class providing the functionalities to handle a meson build system.

    A class providing the functionalities to generate meson build system
    files and compile a meson project.

    Parameters
    ----------
    pyccel_dir : Path
        The directory where generated files should be outputted.
    root_dir : Path
        The directory from which the pyccel-make command was called.
    output_dir : Path
        The directory where the final files should be outputted.
    verbose : int
        The level of verbosity.
    debug_mode : bool
        Indicates if we should compile in debug mode.
    compiler : pyccel.codegen.compilers.compiling.Compiler
        The compiler that should be used to compile the code.
    accelerators : iterable[str]
        Tool used to accelerate the code (e.g., OpenMP, OpenACC).
    """

    def _generate_CompileTarget(self, expr):
        """
        Generate the lines describing this CompileTarget for meson.

        Generate the lines of a meson.build file which describe the
        compilation of this CompileTarget. If the CompileTarget has an
        associated Python wrapper or executable the targets for these
        objects are also described.

        Parameters
        ----------
        expr : pyccel.codegen.compilers.compiling.CompileTarget
            The meson target that should be compiled.

        Returns
        -------
        str
            The meson code describing the target(s).
        """
        obj_lib = f'{expr.name}_objs'
        dep_name = f'{expr.name}_dep'
        mod_name = f"'{expr.pyfile.stem}'"

        out_folder_path = self._output_dir / expr.pyfile.parent.relative_to(self._root_dir)
        out_folder = f"'{out_folder_path.as_posix()}'"

        lib_args = [mod_name, f"'{expr.file.name}'", 'build_by_default: false']

        dep_args = ["include_directories: ['.']"]

        deps = {f"{t.name}_dep": None for t in expr.dependencies}
        deps.update({f"{r}_dep": None for r in recognised_libs \
                    if any(d == r or d.startswith(f"{r}/") \
                    for d in expr.stdlib_dependencies)})
        deps['m_dep'] = None
        if 'openmp' in self._accelerators:
            deps['openmp'] = None
        if 'mpi' in self._accelerators:
            deps['mpi'] = None
        if deps:
            dep_args.append(f"dependencies: [{', '.join(deps)}]")

        lib_args += dep_args
        args = ',\n  '.join(lib_args)
        lib_cmd = f'{obj_lib} = static_library({args})\n'

        args = ',\n  '.join(chain([f'objects: {obj_lib}.extract_all_objects(recursive: true)'], dep_args))
        dep_cmd = f'{dep_name} = declare_dependency(\n  {args})'

        ext_std_deps = {f"{r}_dep": None for r in recognised_libs \
                    if any(d == r or d.startswith(f"{r}/") \
                    for deps in expr.wrapper_files.values() for d in deps)}
        ext_deps = ', '.join((dep_name, *ext_std_deps))

        args = ',\n  '.join([mod_name,
                             *(f"'{w.name}'" for w in expr.wrapper_files),
                             f"dependencies: [{ext_deps}]",
                              "install: true",
                             f"install_dir: {out_folder}",
                             "install_rpath: join_paths(get_option('prefix'), get_option('libdir'))"])
        wrap_cmd = f'py.extension_module({args})\n'

        cmds = [lib_cmd, dep_cmd, wrap_cmd]

        if expr.is_exe:
            args = ',\n  '.join((mod_name, f"'{expr.program_file.name}'",
                                 f"dependencies: [{dep_name}]",
                                  "install: true",
                                 f"install_dir: {out_folder}"))
            cmds.append(f'prog_{expr.name} = executable({args})\n')

        return '\n\n'.join(cmds)

    def _generate_DirTarget(self, expr):
        """
        Generate the meson.build file describing the targets in a directory.

        Generate the meson.build file for a directory. This file describes all
        targets in the directory and imports any sub-directories.

        Parameters
        ----------
        expr : DirTarget
            The Pyccel description of the targets in the directory.

        Returns
        -------
        code : str
            The code that should be printed in the meson.build file to describe
            the targets in this directory.
        include_code : str
            The string that should be printed in the meson.build file found in
            the enclosing directory, to include this new file.
        """
        targets = []
        # Visit the compilation or sub-directory targets found in the directory
        for t in expr.targets:
            if isinstance(t, DirTarget):
                code, subdir_cmd = self._generate_DirTarget(t)
                filename = self._pyccel_dir / t.folder.relative_to(self._root_dir) / 'meson.build'

                if self._verbose > 1:
                    print(">>> Printing :: ", filename)

                # Print sub-directory meson.build file
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(code)
                # Add sub-directory import command to current meson.build file
                targets.append(subdir_cmd)
            else:
                targets.append(self._generate_CompileTarget(t))

        code = '\n'.join(targets)

        return code, f"subdir('{expr.folder.stem}')\n"

    def generate(self, expr):
        """
        Generate all meson.build files necessary to describe the project.

        Generate all meson.build files necessary to describe the project to
        meson. With these files it should be possible to compile the project
        using meson.

        Parameters
        ----------
        expr : BuildProject
            A BuildProject object describing all necessary build information
            for the project.
        """
        languages = ', '.join(f"'{l}'" for l in expr.languages)
        project_decl = f"project('{expr.project_name}', {languages}, meson_version: '>=1.1.0')\n"

        # Python dependencies
        py_import = f"py = import('python').find_installation('{Path(sys.executable).as_posix()}', modules: ['numpy'])\n"
        math_dep = "cc = meson.get_compiler('c')\nm_dep = cc.find_library('m', required : false)\n"
        py_deps = ''.join(("# Python dependencies\n", py_import, math_dep))

        sections = [project_decl, py_deps]

        if 'openmp' in self._accelerators:
            sections.append("openmp = dependency('openmp')")

        if 'mpi' in self._accelerators:
            sections.append("mpi = dependency('mpi')")

        for d in expr.stdlib_deps:
            lib_install = recognised_libs.get(d, None)
            if isinstance(lib_install, ExternalLibInstaller):
                dep_str = f"{d}_dep = dependency('{lib_install.name}'"
                if lib_install.discovery_method == 'CMake':
                    dep_str += f", method : 'cmake', modules : ['{lib_install.name}::{lib_install.target_name}']"
                dep_str += ")\n"
                sections.append(dep_str)
            else:
                sections.append(f"subdir('{d}')\n")

        if 'gFTL_extensions' in expr.stdlib_deps:
            gFTL_extensions_obj = expr.stdlib_deps['gFTL_extensions']
            folder = next(iter(gFTL_extensions_obj.values())).source_folder
            with open(folder / 'meson.build', 'w', encoding='utf-8') as f:
                f.write("gFTL_extensions_mod = static_library('gFTL_extensions',\n")
                for file in gFTL_extensions_obj:
                    f.write(f"    '{file.split('/')[-1]}.F90',\n")
                f.write('    dependencies: [gFTL_dep, gFTL_functions_dep]\n')
                f.write(')\n')
                f.write("gFTL_extensions_dep = declare_dependency(link_with: gFTL_extensions_mod)\n")

        target_code, _ = self._generate_DirTarget(expr.dir_info)

        sections.append(target_code)

        code = '\n'.join(sections)

        filename = self._pyccel_dir / 'meson.build'
        if self._verbose > 1:
            print(">>> Printing :: ", filename)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(code)

    def compile(self):
        """
        Use meson to compile the project.

        Use meson to compile the project.
        """
        capture_output = (self._verbose == 0)
        meson = shutil.which('meson')
        buildtype = 'debug' if self._debug_mode else 'release'

        if self._verbose:
            print(">> Running meson")

        setup_cmd = [meson, 'setup', 'build', '--buildtype', buildtype, '--prefix', str(self._pyccel_dir / 'install')]
        if self._verbose > 1:
            print(" ".join(setup_cmd))
        env = os.environ.copy()
        env['CC'] = self._compiler.get_exec((), 'c')
        env['FC'] = self._compiler.get_exec((), 'fortran')
        subprocess.run(setup_cmd, check=True, cwd=self._pyccel_dir,
                       capture_output=capture_output, env = env)

        build_cmd = [meson, 'compile', '-C', 'build']
        if self._verbose > 1:
            print(" ".join(build_cmd))
        subprocess.run(build_cmd, check=True, cwd=self._pyccel_dir,
                       capture_output=capture_output, env = os.environ)

        install_cmd = [meson, 'install', '-C', 'build']
        if self._verbose > 1:
            print(" ".join(install_cmd))
        subprocess.run(install_cmd, check=True, cwd=self._pyccel_dir,
                       capture_output=capture_output, env = os.environ)
