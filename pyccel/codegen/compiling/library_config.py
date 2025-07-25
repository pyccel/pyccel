# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
This module contains tools useful for handling the compilation of stdlib imports.
"""
import filecmp
import os
from pathlib import Path
import platform
import shutil
import subprocess

from filelock import FileLock

from .basic import CompileObj

from pyccel.ast.numpy_wrapper                    import get_numpy_max_acceptable_version_file

import pyccel.extensions as ext_folder
import pyccel.stdlib as stdlib_folder

#------------------------------------------------------------------------------------------

# get path to pyccel/stdlib/lib_name
stdlib_path = Path(stdlib_folder.__file__).parent

# get path to pyccel/extensions_install/lib_name
ext_path = Path(ext_folder.__file__).parent

#------------------------------------------------------------------------------------------

class StdlibCompileObj:
    def __init__(self, file_name, folder, dependencies = (), **kwargs):
        self._src_dir = stdlib_path / folder
        self._file_name = file_name
        self._folder = folder
        self._dependencies = dependencies
        self._compile_obj_kwargs = kwargs
        assert 'include' not in kwargs
        assert 'libdir' not in kwargs

    def install_to(self, pyccel_dirpath, already_installed):
        """
        Install the files to the Pyccel dirpath.

        Install the files to the Pyccel dirpath so they can be easily located and analysed by
        users. This function copies the contents of the source folder unless the folder already
        exists with the same contents.

        Parameters
        ----------
        pyccel_dirpath : str | Path
            The path to the Pyccel working directory where the copy should be created.

        Returns
        -------
        CompileObj
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
                src_files = list(self._src_dir.glob('*'))
                match, mismatch, errs = filecmp.cmpfiles(self._src_dir, lib_dest_path, src_files)
                to_copy = len(mismatch) != 0
                to_delete = to_copy

            if to_delete:
                os.rmtree(lib_dest_path)

            if to_copy:
                # Copy all files from the source to the destination
                shutil.copytree(self._src_dir, lib_dest_path)

        dependencies = []
        for d in self._dependencies:
            if d in already_installed:
                dependencies.append(already_installed[d])
            else:
                dependencies.append(recognised_libs[d].install_to(pyccel_dirpath, already_installed))

        return CompileObj(self._file_name, lib_dest_path, dependencies = dependencies,
                          **self._compile_obj_kwargs)

class CWrapperCompileObj(StdlibCompileObj):
    def install_to(self, pyccel_dirpath, already_installed):
        super().install_to(pyccel_dirpath, already_installed)
        numpy_file = self.source_folder / 'numpy_version.h'
        with open(numpy_file, 'w') as f:
            f.writelines(get_numpy_max_acceptable_version_file())

#------------------------------------------------------------------------------------------

class ExternalCompileObj:
    def __init__(self, dest_dir, src_dir = None):
        src_dir = src_dir or dest_dir
        self._src_dir = ext_path / src_dir
        self._dest_dir = dest_dir

    def _check_for_package(self, pkg_name):
        pkg_config = shutil.which('pkg-config')
        if not pkg_config:
            return None

        p = subprocess.run([pkg_config, pkg_name], env = os.environ, capture_output = True)
        if p.returncode != 0:
            return None

        p = subprocess.run([pkg_config, pkg_name, '--cflags-only-I'], capture_output = True, text = True)
        include = {i.removeprefix('-I') for i in p.stdout.split()}

        p = subprocess.run([pkg_config, pkg_name, '--cflags-only-other'], capture_output = True, text = True)
        flags = list(p.stdout.split())

        p = subprocess.run([pkg_config, pkg_name, '--libs-only-L'], capture_output = True, text = True)
        libdir = {l.removeprefix('-L') for l in p.stdout.split()}

        p = subprocess.run([pkg_config, pkg_name, '--libs-only-l'], capture_output = True, text = True)
        libs = list(p.stdout.split())

        p = subprocess.run([pkg_config, pkg_name, '--libs-only-other'], capture_output = True, text = True)
        assert p.stdout.strip() == ''

        return CompileObj(pkg_name, folder = "", has_target_file = False,
                          include = include, flags = flags, libdir = libdir,
                          lib = libs)

#------------------------------------------------------------------------------------------

class STCCompileObj(ExternalCompileObj):
    def __init__(self):
        super().__init__("STC")
        self._compile_obj = CompileObj("stc", folder = self._src_dir, has_target_file = False,
                                       include = ("include",), libdir = ("lib/*",))


    def install_to(self, pyccel_dirpath, already_installed, is_debug = False, compiler_family = None):
        """
        Returns
        -------
        CompileObj
        """
        if compiler_family is None:
            compiler_family = 'gnu'

        self._lock_source  = FileLock(str(pyccel_dirpath / 'STC.lock'))

        existing_installation = self._check_for_package('stc')
        if existing_installation:
            return existing_installation

        meson = shutil.which('mesons')
        ninja = shutil.which('ninja')
        has_meson = meson is not None and ninja is not None
        build_dir = pyccel_dirpath / 'STC' / f'build-{compiler_family}'
        install_dir = pyccel_dirpath / 'STC' / 'install'
        with FileLock(install_dir.with_suffix('.lock')):
            if not build_dir.exists():
                if has_meson:
                    buildtype = 'debug' if is_debug else 'release'
                    subprocess.run([meson, 'setup', build_dir, '--buildtype', buildtype, '--prefix', install_dir],
                                    check=True, cwd=self._src_dir)
                    subprocess.run([meson, 'compile', '-C', build_dir], check=True, cwd=pyccel_dirpath)
                    subprocess.run([meson, 'install', '-C', build_dir], check=True, cwd=pyccel_dirpath)
                else:
                    make = shutil.which('make')
                    sh = shutil.which('sh')
                    libdir = install_dir / 'lib' / f'{platform.machine()}-{platform.system().lower()}-{compiler_family}'
                    incdir = install_dir / 'include'
                    os.makedirs(install_dir)
                    os.makedirs(libdir)
                    os.makedirs(libdir / 'pkgconfig')
                    subprocess.run([make, 'lib', f'BUILDDIR={build_dir}', '-C', self._src_dir],
                                   check=True, cwd=pyccel_dirpath)
                    shutil.copytree(ext_path / 'STC' / 'include', incdir)
                    shutil.copyfile(build_dir / 'libstc.a', libdir / 'libstc.a')
                    with open(libdir / 'pkgconfig' / 'stc.pc', 'w', encoding='utf-8') as f:
                        f.write("Name: stc\n")
                        f.write("Description: stc\n")
                        f.write("Version: 5.0-dev\n")
                        f.write(f"Libs: -L{libdir} -lstc -lm\n")
                        f.write(f"Cflags: -I{incdir}")

        libdir = next((install_dir / 'lib').glob(f'*-{compiler_family}'))

        include = {install_dir / 'include'}
        libdir = {libdir}
        libs = ['-lstc', '-lm']

        return CompileObj("stc", folder = "", has_target_file = False,
                          include = include, libdir = libdir, lib = libs)

#------------------------------------------------------------------------------------------

class GFTLCompileObj(ExternalCompileObj):
    def __init__(self):
        super().__init__("gFTL", src_dir = "gFTL/install/GFTL-1.13")

    def install_to(self, pyccel_dirpath, already_installed):
        """
        Install the files to the Pyccel dirpath.

        Install the files to the Pyccel dirpath so they can be easily located and analysed by
        users. This function creates a symlink as the contents of this folder are not expected
        to change.

        Parameters
        ----------
        pyccel_dirpath : str | Path
            The path to the Pyccel working directory where the link should be created.

        Returns
        -------
        CompileObj
        """
        dest_dir = Path(pyccel_dirpath) / self._dest_dir
        if not dest_dir.exists():
            os.symlink(self._src_dir, dest_dir, target_is_directory=True)

        self._lock_source  = FileLock(str(pyccel_dirpath / 'gFTL.lock'))

        return CompileObj("gFTL", folder = "gFTL", has_target_file = False,
                          include = (dest_dir / 'include/v2',))

#------------------------------------------------------------------------------------------

recognised_libs = {
    # External libs
    "stc"  : STCCompileObj(),
    "gFTL" : GFTLCompileObj(),
    # Internal libs
    "pyc_math_f90"   : StdlibCompileObj("pyc_math_f90.f90", "math", libs = ('m',)),
    "pyc_math_c"     : StdlibCompileObj("pyc_math_c.c", "math"),
    "pyc_tools_f90"  : StdlibCompileObj("pyc_tools_f90.f90", "tools"),
    "cwrapper"       : CWrapperCompileObj("cwrapper.c", "cwrapper", accelerators=('python',)),
    "STC_Extensions" : StdlibCompileObj("STC_Extensions", "STC_Extensions",
                                        has_target_file = False,
                                        dependencies = ('stc',)),
    "gFTL_functions" : StdlibCompileObj("*.inc", "gFTL_functions",
                                        has_target_file = False,
                                        dependencies = ('gFTL',))
}

recognised_libs['CSpan_extensions'] = recognised_libs['STC_Extensions']
