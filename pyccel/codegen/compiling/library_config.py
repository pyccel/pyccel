# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
This module contains tools useful for handling the compilation of stdlib imports.
"""
import filecmp
from itertools import chain
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from filelock import FileLock

from pyccel.ast.numpy_wrapper                    import get_numpy_max_acceptable_version_file
from pyccel.errors.errors import Errors

import pyccel.extensions as ext_folder
import pyccel.stdlib as stdlib_folder

from .basic import CompileObj

#------------------------------------------------------------------------------------------

errors = Errors()

# get path to pyccel/stdlib/lib_name
stdlib_path = Path(stdlib_folder.__file__).parent

# get path to pyccel/extensions_install/lib_name
ext_path = Path(ext_folder.__file__).parent

#------------------------------------------------------------------------------------------

class StdlibInstaller:
    """
    A class describing how stdlib objects are installed.

    A class describing how stdlib objects are installed. An Installer has a `install_to`
    method which creates a CompileObj that can be used as a dependency in translations.

    Parameters
    ----------
    file_name : str
        Name of file that will be compiled.
    folder : str
        Name of the folder in the stdlib folder where the file is found.
    dependencies : iterable[str], optional
        An iterable containing the names of all the (external or internal) libraries
        on which this internal library depends.
    **kwargs : dict
        A dictionary of additional keyword arguments that will be used when creating
        the CompileObj. See CompileObj for more details.
    """
    def __init__(self, file_name, folder, dependencies = (), **kwargs):
        self._src_dir = stdlib_path / folder
        self._file_name = file_name
        self._folder = folder
        self._dependencies = dependencies
        self._compile_obj_kwargs = kwargs
        assert 'include' not in kwargs
        assert 'libdir' not in kwargs

    def install_to(self, pyccel_dirpath, installed_libs, verbose, compiler):
        """
        Install the files to the Pyccel dirpath.

        Install the files to the Pyccel dirpath so they can be easily located and analysed by
        users. This function copies the contents of the source folder unless the folder already
        exists with the same contents. It returns the CompileObj that describes these new
        files.

        Parameters
        ----------
        pyccel_dirpath : str | Path
            The path to the Pyccel working directory where the copy should be created.
        installed_libs : dict[str, CompileObj]
            A dictionary describing all the libraries that have already been installed. This
            ensures that new CompileObjs are not created if multiple objects share the same
            library dependencies.
        verbose : int
            The level of verbosity.
        compiler : pyccel.codegen.compilers.compiling.Compiler
            A Compiler object in case the installed dependency needs compiling. This is
            unused in this method.

        Returns
        -------
        CompileObj
            The object that should be added as a dependency to objects that depend on this
            library.
        """
        lib_dest_path = pyccel_dirpath / self._folder
        lock  = FileLock(str(lib_dest_path.with_suffix('.lock')))
        with lock:
            # Check if folder exists
            if not lib_dest_path.exists():
                to_copy = True
                to_delete = False
            else:
                # If folder exists check if it needs updating
                src_files = [f.relative_to(self._src_dir) for f in self._src_dir.glob('*')]
                _, mismatch, _ = filecmp.cmpfiles(lib_dest_path, self._src_dir, src_files)
                to_copy = len(mismatch) != 0
                to_delete = to_copy

            if to_delete:
                shutil.rmtree(lib_dest_path)

            if to_copy:
                if verbose:
                    print(f">> Copying {self._src_dir} to {lib_dest_path}")
                # Copy all files from the source to the destination
                shutil.copytree(self._src_dir, lib_dest_path)

        dependencies = []
        for d in self._dependencies:
            if d in installed_libs:
                dependencies.append(installed_libs[d])
            else:
                dependencies.append(recognised_libs[d].install_to(pyccel_dirpath, installed_libs, verbose, compiler))

        new_obj = CompileObj(self._file_name, lib_dest_path, dependencies = dependencies,
                          **self._compile_obj_kwargs)
        installed_libs[self._folder] = new_obj
        return new_obj

class CWrapperInstaller(StdlibInstaller):
    """
    A class describing how the cwrapper library is installed.

    A class describing how the cwrapper library is installed. This class inherits from
    StdlibInstaller. The specialisation is required to ensure that the file describing
    the NumPy version is also created.

    Parameters
    ----------
    file_name : str
        Name of file that will be compiled.
    folder : str
        Name of the folder in the stdlib folder where the file is found.
    dependencies : iterable[str], optional
        An iterable containing the names of all the (external or internal) libraries
        on which this internal library depends.
    **kwargs : dict
        A dictionary of additional keyword arguments that will be used when creating
        the CompileObj. See CompileObj for more details.
    """
    def install_to(self, pyccel_dirpath, installed_libs, verbose, compiler):
        """
        Install the files to the Pyccel dirpath.

        Install the files to the Pyccel dirpath so they can be easily located and analysed by
        users. This function copies the contents of the source folder unless the folder already
        exists with the same contents. It returns the CompileObj that describes these new
        files.

        Parameters
        ----------
        pyccel_dirpath : str | Path
            The path to the Pyccel working directory where the copy should be created.
        installed_libs : dict[str, CompileObj]
            A dictionary describing all the libraries that have already been installed. This
            ensures that new CompileObjs are not created if multiple objects share the same
            library dependencies.
        verbose : int
            The level of verbosity.
        compiler : pyccel.codegen.compilers.compiling.Compiler
            A Compiler object in case the installed dependency needs compiling. This is
            unused in this method.

        Returns
        -------
        CompileObj
            The object that should be added as a dependency to objects that depend on this
            library.
        """
        compile_obj = super().install_to(pyccel_dirpath, installed_libs, verbose, compiler)
        numpy_file = compile_obj.source_folder / 'numpy_version.h'
        with open(numpy_file, 'w', encoding='utf-8') as f:
            f.writelines(get_numpy_max_acceptable_version_file())
        return compile_obj

#------------------------------------------------------------------------------------------

class ExternalLibInstaller:
    """
    A class describing how external libraries used by Pyccel are installed.

    A class describing how external libraries used by Pyccel are installed. An Installer
    has a `install_to` method which creates a CompileObj that can be used as a dependency in translations.

    Parameters
    ----------
    dest_dir : str
        The name of the sub-folder into which the library should be installed. This
        decides the name of the folder that will be created in the `__pyccel__` folder.
    src_dir : str, optional
        The name of the sub-folder where the library can be found in the extensions/ folder.
        The default is to use the same as the `dest_dir` parameter.
    """
    def __init__(self, dest_dir, src_dir = None):
        src_dir = src_dir or dest_dir
        self._src_dir = ext_path / src_dir
        self._dest_dir = dest_dir
        self._discovery_method = None

    @property
    def discovery_method(self):
        """
        Get the standard method for discovering this package (CMake vs pkgconfig).

        Get the standard method for discovering this package (CMake vs pkgconfig). If the
        method is unknown then None is returned. In this case the method should match the
        chosen build system.
        """
        return self._discovery_method

    @property
    def name(self):
        """
        Get the name by which the package is known in the build system.

        Get the name by which the package is known in the build system.
        """
        return self._dest_dir

    def _check_for_cmake_package(self, pkg_name, languages, options = '', *, target_name):
        """
        Use CMake to search for a package.

        Use CMake to search for a package. CMake can provide the compilation
        information.

        Parameters
        ----------
        pkg_name : str
            The name of the package.
        languages : iterable[str]
            The languages that the project will use with this package.
        options : iterable[str], optional
            Any additional options that should be passed to find_package.
            E.g. COMPONENTS.
        target_name : str
            The name of the package target. By default this is assumed to be
            the same as the pkg_name (e.g. HDF5::HDF5).

        Returns
        -------
        CompileObj | None
            A CompileObj describing the package if it is installed on the system.
        """
        cmake = shutil.which('cmake')
        # If cmake is not installed then exit
        if not cmake:
            return None

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as build_dir:
            # Write a minimal CMakeLists.txt
            cmakelists_path = os.path.join(build_dir, "CMakeLists.txt")
            with open(cmakelists_path, "w", encoding='utf-8') as f:
                f.write(f'project(Test LANGUAGES {languages})\n')
                f.write('cmake_minimum_required(VERSION 3.28)\n')
                f.write(f'find_package({pkg_name} REQUIRED {options})\n')
                f.write(f'get_target_property(FLAGS {pkg_name}::{target_name} COMPILE_FLAGS)\n')
                f.write(f'get_target_property(INCLUDE_DIRS {pkg_name}::{target_name} INCLUDE_DIRECTORIES)\n')
                f.write(f'get_target_property(INTERFACE_INCLUDE_DIRS {pkg_name}::{target_name} INTERFACE_INCLUDE_DIRECTORIES)\n')
                f.write(f'get_target_property(LIBRARIES {pkg_name}::{target_name} LINK_LIBRARIES)\n')
                f.write(f'get_target_property(INTERFACE_LIBRARIES {pkg_name}::{target_name} INTERFACE_LINK_LIBRARIES)\n')
                f.write(f'get_target_property(LIB_DIRS {pkg_name}::{target_name} INTERFACE_LINK_DIRECTORIES)\n')
                f.write(f'message(STATUS "{pkg_name} Found : ${{{pkg_name}_FOUND}}")\n')
                f.write('message(STATUS "${FLAGS}")\n')
                f.write('message(STATUS "${INCLUDE_DIRS}")\n')
                f.write('message(STATUS "${INTERFACE_INCLUDE_DIRS}")\n')
                f.write('message(STATUS "${LIBRARIES}")\n')
                f.write('message(STATUS "${INTERFACE_LIBRARIES}")\n')
                f.write('message(STATUS "${LIB_DIRS}")\n')

            # Run cmake configure step in that temp dir
            p = subprocess.run(
                [cmake, "-S", build_dir, "-B", build_dir],
                capture_output=True, text=True, check=False)

        if p.returncode:
            return None
        else:
            self._discovery_method = 'CMake'
            output = p.stdout.split('\n-- ')
            start = next(i for i, l in enumerate(output) if l == f'{pkg_name} Found : 1')
            flags, include_dirs, interface_include_dirs, libs, interface_libs, libdirs = ('' if o.endswith('NOTFOUND') else o for o in output[start+1:start+7])
            return CompileObj(pkg_name, folder = "", has_target_file = False,
                              include = [i for i in chain(include_dirs.split(','),
                                                          interface_include_dirs.split(',')) if i],
                              flags = [f for f in flags.split(',') if f],
                              libdir = [l for l in libdirs.split(',') if l],
                              libs = [l for l in chain(libs.split(','),
                                                       interface_libs.split(',')) if l])

    def _check_for_package(self, pkg_name, options = ()):
        """
        Use pkg-config to search for a package.

        Use pkg-config to search for a package. pkg-config can provide the compilation
        information.

        Parameters
        ----------
        pkg_name : str
            The name of the package.
        options : iterable[str], optional
            Any additional options that should be passed to pkg-config to limit the search.
            E.g. min/max version.

        Returns
        -------
        CompileObj | None
            A CompileObj describing the package if it is installed on the system.
        """
        pkg_config = shutil.which('pkg-config')
        # If pkg-config is not installed then exit
        if not pkg_config:
            return None

        p = subprocess.run([pkg_config, pkg_name, *options], env = os.environ, capture_output = True, check = False)
        # If the package is not found then exit
        if p.returncode != 0:
            return None

        # If the package exists then query pkg-config to get the compilation information
        p = subprocess.run([pkg_config, pkg_name, '--cflags-only-I'], capture_output = True,
                           text = True, check = True)
        include = {i.removeprefix('-I') for i in p.stdout.split()}

        p = subprocess.run([pkg_config, pkg_name, '--cflags-only-other'], capture_output = True,
                           text = True, check = True)
        flags = list(p.stdout.split())

        p = subprocess.run([pkg_config, pkg_name, '--libs-only-L'], capture_output = True,
                           text = True, check = True)
        libdir = {l.removeprefix('-L') for l in p.stdout.split()}

        p = subprocess.run([pkg_config, pkg_name, '--libs-only-l'], capture_output = True,
                           text = True, check = True)
        libs = list(p.stdout.split())

        p = subprocess.run([pkg_config, pkg_name, '--libs-only-other'], capture_output = True,
                           text = True, check = True)
        assert p.stdout.strip() == ''

        self._discovery_method = 'pkgconfig'
        return CompileObj(pkg_name, folder = "", has_target_file = False,
                          include = include, flags = flags, libdir = libdir,
                          libs = libs)

#------------------------------------------------------------------------------------------

class STCInstaller(ExternalLibInstaller):
    """
    A class describing how the external library STC is installed.

    A class describing how the external library STC is installed. This specialisation allows
    the installation procedure to be specialised for this library.
    """
    def __init__(self):
        super().__init__("stc", src_dir = "STC")
        self._compile_obj = CompileObj("stc", folder = self._src_dir, has_target_file = False,
                                       include = ("include",), libdir = ("lib/*",))

    def install_to(self, pyccel_dirpath, installed_libs, verbose, compiler):
        """
        Install the files to the Pyccel dirpath.

        Install the files to the Pyccel dirpath so they can be easily located and analysed by
        users. This function builds and installs the library if it is not already installed.
        It returns the CompileObj that describes the new installation files.

        Parameters
        ----------
        pyccel_dirpath : str | Path
            The path to the Pyccel working directory where the copy should be created.
        installed_libs : dict[str, CompileObj]
            A dictionary describing all the libraries that have already been installed. This
            ensures that new CompileObjs are not created if multiple objects share the same
            library dependencies.
        verbose : int
            The level of verbosity.
        compiler : pyccel.codegen.compilers.compiling.Compiler
            A Compiler object to compile STC if it is not already installed.

        Returns
        -------
        CompileObj
            The object that should be added as a dependency to objects that depend on this
            library.
        """
        compiler_family = compiler.compiler_family.lower()

        # Use pkg-config to try to locate an existing (system or user) installation
        # with version >= 5.0 < 6
        existing_installation = self._check_for_package('stc', ['--max-version=6', '--atleast-version=5'])
        print("EXISTING : ", existing_installation)
        if existing_installation:
            installed_libs['stc'] = existing_installation
            return existing_installation

        # Check if meson can be used to build
        meson = shutil.which('meson')
        ninja = shutil.which('ninja')
        assert meson is not None and ninja is not None
        build_dir = pyccel_dirpath / 'STC' / f'build-{compiler_family}'
        install_dir = pyccel_dirpath / 'STC' / 'install'
        with FileLock(install_dir.with_suffix('.lock')):
            if build_dir.exists() and build_dir.lstat().st_mtime < self._src_dir.lstat().st_mtime:
                print("OLD")
                shutil.rmtree(build_dir)
                shutil.rmtree(install_dir)

            print(build_dir, build_dir.exists())

            # If the build dir already exists then we have already compiled these files
            if not build_dir.exists():
                print("BUILDING")
                buildtype = 'debug' if compiler.is_debug else 'release'
                env = os.environ.copy()
                env['CC'] = compiler.get_exec({}, "c")
                if verbose:
                    print(">> Installing STC with meson")
                p = subprocess.run([meson, 'setup', build_dir, '--buildtype', buildtype, '--prefix', install_dir],
                               check = False, cwd = self._src_dir, env = env,
                               capture_output = (verbose <= 1))
                print(p.stdout)
                print(p.stderr)
                assert p.returncode
                subprocess.run([meson, 'compile', '-C', build_dir], check = True, cwd = pyccel_dirpath,
                               capture_output = (verbose == 0))
                subprocess.run([meson, 'install', '-C', build_dir], check = True, cwd = pyccel_dirpath,
                               capture_output = (verbose <= 1))

        print(install_dir)
        print(list(install_dir.glob('**/*.a')))
        libdir = next(install_dir.glob('**/*.a')).parent
        libs = ['-lstc', '-lm']

        self._discovery_method = 'pkgconfig'
        sep = ';' if sys.platform == "win32" else ':'
        PKG_CONFIG_PATH = os.environ.get('PKG_CONFIG_PATH', '').split(sep)
        os.environ['PKG_CONFIG_PATH'] = ':'.join(p for p in (*PKG_CONFIG_PATH, str(libdir / "pkgconfig"))
                                                 if p and Path(p).exists())

        new_obj = CompileObj("stc", folder = "", has_target_file = False,
                          include = (install_dir / 'include',),
                          libdir = (libdir, ), libs = libs)
        installed_libs['stc'] = new_obj
        return new_obj

#------------------------------------------------------------------------------------------

class GFTLInstaller(ExternalLibInstaller):
    """
    A class describing how the external library gFTL is installed.

    A class describing how the external library gFTL is installed. This specialisation allows
    the installation procedure to be specialised for this library.
    """
    def __init__(self):
        super().__init__("GFTL", src_dir = "gFTL/install")

    @property
    def target_name(self):
        """
        The name of the relevant CMake target inside the gFTL package.

        The name of the relevant CMake target inside the gFTL package.
        """
        return 'gftl-v2'

    def install_to(self, pyccel_dirpath, installed_libs, verbose, compiler):
        """
        Install the files to the Pyccel dirpath.

        Install the files to the Pyccel dirpath so they can be easily located and analysed by
        users. This function creates a symlink to the Pyccel folder containing the code as
        these files are not expected to be modified. The symlink makes it easier for users to
        examine the code used. The CompileObj that describes the files is returned.

        Parameters
        ----------
        pyccel_dirpath : str | Path
            The path to the Pyccel working directory where the copy should be created.
        installed_libs : dict[str, CompileObj]
            A dictionary describing all the libraries that have already been installed. This
            ensures that new CompileObjs are not created if multiple objects share the same
            library dependencies.
        verbose : int
            The level of verbosity.
        compiler : pyccel.codegen.compilers.compiling.Compiler
            A Compiler object in case the installed dependency needs compiling. This is
            unused in this method.

        Returns
        -------
        CompileObj
            The object that should be added as a dependency to objects that depend on this
            library.
        """
        existing_installation = self._check_for_cmake_package('GFTL', 'Fortran', target_name = self.target_name)
        if existing_installation:
            installed_libs['gFTL'] = existing_installation
            return existing_installation
        dest_dir = Path(pyccel_dirpath) / self._dest_dir
        if not dest_dir.exists():
            if verbose:
                print(f">> Creating a link to {self._src_dir} in {dest_dir}")
            os.symlink(self._src_dir, dest_dir, target_is_directory=True)

        new_obj = CompileObj("gFTL", folder = "gFTL", has_target_file = False,
                          include = (dest_dir / 'GFTL-1.13/include/v2',))
        installed_libs['gFTL'] = new_obj

        self._discovery_method = 'CMake'
        sep = ';' if sys.platform == "win32" else ':'
        CMAKE_PREFIX_PATH = os.environ.get('CMAKE_PREFIX_PATH', '').split(sep)
        os.environ['CMAKE_PREFIX_PATH'] = ':'.join(s for s in (*CMAKE_PREFIX_PATH, str(dest_dir))
                                                   if s and Path(s).exists())

        return new_obj

#------------------------------------------------------------------------------------------

recognised_libs = {
    # External libs
    "stc"  : STCInstaller(),
    "gFTL" : GFTLInstaller(),
    # Internal libs
    "pyc_math_f90"   : StdlibInstaller("pyc_math_f90.f90", "math", libs = ('m',)),
    "pyc_math_c"     : StdlibInstaller("pyc_math_c.c", "math"),
    "pyc_tools_f90"  : StdlibInstaller("pyc_tools_f90.f90", "tools"),
    "cwrapper"       : CWrapperInstaller("cwrapper.c", "cwrapper", extra_compilation_tools=('python',)),
    "STC_Extensions" : StdlibInstaller("STC_Extensions", "STC_Extensions",
                                        has_target_file = False,
                                        dependencies = ('stc',)),
    "gFTL_functions" : StdlibInstaller("gFTL_functions", "gFTL_functions",
                                        has_target_file = False,
                                        dependencies = ('gFTL',)),
    "gFTL_extensions" : None
}

recognised_libs['CSpan_extensions'] = recognised_libs['STC_Extensions']
