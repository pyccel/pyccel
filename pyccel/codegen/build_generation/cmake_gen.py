"""
A module to handle everything related to CMake for the pyccel-make command.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from pyccel.codegen.compiling.project import DirTarget
from pyccel.codegen.compiling.library_config import recognised_libs, ExternalLibInstaller

from .build_gen import BuildSystemHandler

class CMakeHandler(BuildSystemHandler):
    """
    A class providing the functionalities to handle a CMake build system.

    A class providing the functionalities to generate CMake build system
    files and compile a CMake project.

    Parameters
    ----------
    pyccel_dir : Path
        The directory where generated files should be outputted.
    root_dir : Path
        The directory from which the pyccel-make command was called.
    verbose : int
        The level of verbosity.
    debug_mode : bool
        Indicates if we should compile in debug mode.
    compiler : Compiler
        The compiler that should be used to compile the code.
    accelerators : iterable[str]
        Tool used to accelerate the code (e.g., OpenMP, OpenACC).
    """
    def __init__(self, *args, **kwargs):
        cmake = shutil.which('cmake')
        with tempfile.TemporaryDirectory() as build_dir:
            # Write a minimal CMakeLists.txt
            cmakelists_path = os.path.join(build_dir, "CMakeLists.txt")
            with open(cmakelists_path, "w", encoding='utf-8') as f:
                f.write(f'project(Test LANGUAGES C)\n')
                f.write('cmake_minimum_required(VERSION 3.28)\n')
                f.write(f'find_library(MATH_LIBRARY m REQUIRED)\n')

            # Run cmake configure step in that temp dir
            p = subprocess.run(
                [cmake, "-S", build_dir, "-B", build_dir],
                capture_output=True, text=True, check=False)

        self._math_lib_available_on_platform = p.returncode == 0

        super().__init__(*args, **kwargs)

    def _generate_CompileTarget(self, expr):
        """
        Generate the lines describing this CompileTarget for CMake.

        Generate the lines of a CMakeLists.txt file which describe the
        compilation of this CompileTarget. If the CompileTarget has an
        associated Python wrapper or executable the targets for these
        objects are also described.

        Parameters
        ----------
        expr : CompileTarget
            The CMake target that should be compiled.

        Returns
        -------
        str
            The CMake code describing the target(s).
        """
        kernel_target = expr.name
        mod_name = expr.pyfile.stem

        out_folder = expr.pyfile.parent

        args = '\n    '.join([kernel_target, 'STATIC', expr.file.name])
        cmds = [f'add_library({args})\n']

        to_link = {t.name for t in expr.dependencies}
        to_link.update(r for r in recognised_libs \
                    if any(d == r or d.startswith(f"{r}/") \
                    for d in expr.stdlib_dependencies))
        if self._math_lib_available_on_platform:
            to_link.add('${MATH_LIBRARY}')
        if expr.file.suffix == '.f90':
            args = '\n    '.join([kernel_target, 'PUBLIC', '${CMAKE_CURRENT_BINARY_DIR}', '${CMAKE_CURRENT_SOURCE_DIR}'])
            cmds.append(f'target_include_directories({args})\n')
            if 'openmp' in self._accelerators:
                to_link.add('OpenMP::OpenMP_Fortran')
            if 'mpi' in self._accelerators:
                to_link.add('MPI::MPI_Fortran')
        else:
            args = '\n    '.join([kernel_target, 'PUBLIC', '${CMAKE_CURRENT_SOURCE_DIR}'])
            cmds.append(f'target_include_directories({args})\n')
            if 'openmp' in self._accelerators:
                to_link.add('OpenMP::OpenMP_C')
            if 'mpi' in self._accelerators:
                to_link.add('MPI::MPI_C')

        if to_link:
            link_args = '\n    '.join([kernel_target, 'PUBLIC', *to_link])
            cmds.append(f"target_link_libraries({link_args})\n")


        wrap_args = '\n    '.join([f'{kernel_target}_so', 'MODULE', 'WITH_SOABI',
                                    *[w.name for w in expr.wrapper_files]])
        cmds.append(f'Python_add_library({wrap_args})\n')

        target_args = '\n    '.join([f'{kernel_target}_so', 'PROPERTIES', 'OUTPUT_NAME', mod_name])
        cmds.append(f'set_target_properties({target_args})')

        ext_std_deps = {r: None for r in recognised_libs \
                    if any(d == r or d.startswith(f"{r}/") \
                    for deps in expr.wrapper_files.values() for d in deps)}
        link_args = '\n    '.join([f'{kernel_target}_so', 'PUBLIC', kernel_target, 'cwrapper', *ext_std_deps])
        cmds.append(f"target_link_libraries({link_args})\n")
        args = '\n    '.join([f'{kernel_target}_so', 'PUBLIC', '${CMAKE_CURRENT_SOURCE_DIR}'])
        cmds.append(f'target_include_directories({args})\n')

        args = '\n    '.join(['TARGETS', f'{kernel_target}_so', 'DESTINATION', out_folder.as_posix()])
        cmds.append(f"install({args})\n")

        if expr.is_exe:
            prog_target = f'prog_{expr.name}'
            args = '\n    '.join([prog_target, expr.program_file.name])
            cmds.append(f'add_executable({args})\n')

            args = '\n    '.join([prog_target, 'PROPERTIES', 'OUTPUT_NAME', mod_name])
            cmds.append(f'set_target_properties({args})\n')

            args = '\n    '.join([prog_target, 'PUBLIC', kernel_target])
            cmds.append(f'target_link_libraries({args})')

            args = '\n    '.join(['TARGETS', prog_target, 'DESTINATION', out_folder.as_posix()])
            cmds.append(f"install({args})\n")

        return '\n'.join(cmds)

    def _generate_DirTarget(self, expr):
        """
        Generate the CMakeLists.txt file describing the targets in a directory.

        Generate the CMakeLists.txt file for a directory. This file describes all
        targets in the directory and imports any sub-directories.

        Parameters
        ----------
        expr : DirTarget
            The Pyccel description of the targets in the directory.

        Returns
        -------
        str
            The string that should be printed in the enclosing CMakeLists.txt file
            to include this new file.
        """
        targets = []
        for t in expr.targets:
            if isinstance(t, DirTarget):
                code, subdir_cmd = self._generate_DirTarget(t)
                filename = self._pyccel_dir / t.folder.relative_to(self._root_dir) / 'CMakeLists.txt'

                if self._verbose > 1:
                    print(">>> Printing :: ", filename)

                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(code)
                targets.append(subdir_cmd)
            else:
                targets.append(self._generate_CompileTarget(t))

        code = '\n'.join(targets)
        return code, f"add_subdirectory({expr.folder.stem})\n"

    def generate(self, expr):
        """
        Generate all CMakeLists.txt files necessary to describe the project.

        Generate all CMakeLists.txt files necessary to describe the project to
        CMake. With these files it should be possible to compile the project
        using CMake.

        Parameters
        ----------
        expr : BuildProject
            A BuildProject object describing all necessary build information
            for the project.
        """
        cmake_min = 'cmake_minimum_required(VERSION 3.20)'

        languages = ' '.join(['LANGUAGES', *[l.capitalize() for l in expr.languages]])
        project_decl = f"project('{expr.project_name}' {languages})\n"

        pic_on = 'set(CMAKE_POSITION_INDEPENDENT_CODE ON)\n'

        # Python dependencies
        version = sys.version_info

        py_import = (f'set(Python_ROOT_DIR {Path(sys.executable).parent.parent.as_posix()})\n'
                     f"find_package(Python {version.major}.{version.minor}.{version.micro} EXACT REQUIRED COMPONENTS Development NumPy)\n")

        math_import = ''
        if self._math_lib_available_on_platform:
            math_import = 'find_library(MATH_LIBRARY m)\n'

        sections = [cmake_min, project_decl, pic_on, py_import, math_import]

        if 'openmp' in self._accelerators:
            sections.append('find_package(OpenMP REQUIRED)\n')

        if 'mpi' in self._accelerators:
            sections.append('find_package(MPI REQUIRED)\n')

        pkg_config_needed = False
        for folder in expr.stdlib_deps:
            lib_install = recognised_libs.get(folder, None)
            if isinstance(lib_install, ExternalLibInstaller):
                if lib_install.discovery_method == 'pkgconfig':
                    pkg_config_needed = True
                    sections.append((f"pkg_check_modules({folder} REQUIRED IMPORTED_TARGET {folder})\n"
                                     f"add_library({folder} ALIAS PkgConfig::{folder})\n"))
                else:
                    sections.append(f"find_package({lib_install.name} REQUIRED)\n")
            else:
                sections.append(f"add_subdirectory({folder})\n")

        if 'gFTL_extensions' in expr.stdlib_deps:
            gFTL_extensions_obj = expr.stdlib_deps['gFTL_extensions']
            folder = next(iter(gFTL_extensions_obj.values())).source_folder
            with open(folder / 'CMakeLists.txt', 'w', encoding='utf-8') as f:
                f.write('add_library(gFTL_extensions\n    STATIC\n')
                for file in gFTL_extensions_obj:
                    f.write(f"    {file.split('/')[-1]}.F90\n")
                f.write(')\n')
                f.write('target_include_directories(gFTL_extensions\n')
                f.write('    PUBLIC "${CMAKE_CURRENT_BINARY_DIR}"\n')
                f.write(')\n\n')
                f.write('target_link_libraries(gFTL_extensions\n')
                f.write('    PUBLIC\n')
                f.write('    gFTL_functions\n')
                f.write(f'    GFTL::{recognised_libs["gFTL"].target_name}\n')
                f.write(')\n')

        if pkg_config_needed:
            sections.insert(4, "find_package(PkgConfig REQUIRED)\n")

        target_code, _ = self._generate_DirTarget(expr.dir_info)

        sections.append(target_code)

        code = '\n'.join(sections)

        filename = self._pyccel_dir / 'CMakeLists.txt'
        if self._verbose > 1:
            print(">>> Printing :: ", filename)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(code)

    def compile(self):
        """
        Use CMake to compile the project.

        Use CMake to compile the project.
        """
        capture_output = (self._verbose == 0)
        cmake = shutil.which('cmake')
        buildtype = 'Debug' if self._debug_mode else 'Release'

        if self._verbose:
            print(">> Running CMake")

        setup_cmd = [cmake, '-B', str(self._pyccel_dir / 'build'),
                     f'-DCMAKE_BUILD_TYPE={buildtype}', '-S', str(self._pyccel_dir)]
        if sys.platform == 'win32':
            setup_cmd.append('-G')
            setup_cmd.append('MinGW Makefiles')

        if self._verbose > 1:
            print(" ".join(setup_cmd))
        env = os.environ.copy()
        env['CC'] = self._compiler.get_exec(self._accelerators, 'c')
        env['FC'] = self._compiler.get_exec(self._accelerators, 'fortran')
        subprocess.run(setup_cmd, check=True, env=env,
                       capture_output=capture_output)

        build_cmd = [cmake, '--build', str(self._pyccel_dir / 'build')]
        if self._verbose > 1:
            print(" ".join(build_cmd))
        subprocess.run(build_cmd, check=True, cwd=self._pyccel_dir,
                       capture_output=capture_output)

        install_cmd = [cmake, '--install', str(self._pyccel_dir / 'build')]
        if self._verbose > 1:
            print(" ".join(install_cmd))
        subprocess.run(install_cmd, check=True, cwd=self._pyccel_dir,
                       capture_output=capture_output)
