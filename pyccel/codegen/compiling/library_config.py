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
    def __init__(self, file_name, folder, include = (), libdir = (), **kwargs):
        self._src_dir = stdlib_path / folder
        self._file_name = file_name
        self._include = include
        self._libdir = libdir
        self._folder = folder
        self._compile_obj_kwargs = {**kwargs}

    def install_to(self, pyccel_dirpath):
        """
        Install the files to the Pyccel dirpath.

        Install the files to the Pyccel dirpath so they can be easily located and analysed by
        users. This function copies the contents of the source folder unless the folder already
        exists with the same contents.

        Parameters
        ----------
        pyccel_dirpath : str | Path
            The path to the Pyccel working directory where the copy should be created.
        """
        lib_dest_path = pyccel_dirpath / self._folder
        self._compile_obj = CompileObj(self._file_name, folder = lib_dest_path, **self._compile_obj_kwargs)
        with FileLock(lib_dest_path.with_suffix('.lock')):
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

    @property
    def compile_obj(self):
        return self._compile_obj

class CWrapperCompileObj(StdlibCompileObj):
    def install_to(self, pyccel_dirpath):
        super().install_to(pyccel_dirpath)
        numpy_file = self._compile_obj.source_folder / 'numpy_version.h'
        with open(numpy_file, 'w') as f:
            f.writelines(get_numpy_max_acceptable_version_file())

#------------------------------------------------------------------------------------------

class ExternalCompileObj:
    def __init__(self, dest_dir, src_dir = None):
        src_dir = src_dir or dest_dir
        self._src_dir = ext_path / src_dir
        self._dest_dir = dest_dir

    @property
    def dependency(self):
        return self._compile_obj

    @property
    def compile_obj(self):
        return self._compile_obj

    def _check_for_package(self, pkg_name):
        pkg_config = shutil.which('pkg-config')
        if not pkg_config:
            return False

        p = subprocess.run([pkg_config, pkg_name], env = os.environ)
        if p.returncode != 0:
            return False

        p = subprocess.run([pkg_config, pkg_name, '--cflags-only-I'], capture_output = True, text = True)
        self._compile_obj.include = {i.removeprefix('-I') for i in p.stdout.split()}

        p = subprocess.run([pkg_config, pkg_name, '--cflags-only-other'], capture_output = True, text = True)
        self._compile_obj.flags = list(p.stdout.split())

        p = subprocess.run([pkg_config, pkg_name, '--libs-only-L'], capture_output = True, text = True)
        self._compile_obj.libdir = {l.removeprefix('-L') for l in p.stdout.split()}

        p = subprocess.run([pkg_config, pkg_name, '--libs-only-l'], capture_output = True, text = True)
        self._compile_obj.libs = list(p.stdout.split())

        p = subprocess.run([pkg_config, pkg_name, '--libs-only-other'], capture_output = True, text = True)
        assert p.stdout.strip() == ''

        return True

#------------------------------------------------------------------------------------------

class STCCompileObj(ExternalCompileObj):
    def __init__(self):
        super().__init__("STC")
        self._compile_obj = CompileObj("stc", folder = self._src_dir, has_target_file = False,
                                       include = ("include",), libdir = ("lib/*",))


    def install_to(self, pyccel_dirpath, is_debug = False, compiler = None):
        if compiler is None:
            compiler_family = 'gnu'

        if self._check_for_package('stc'):
            return

        meson = shutil.which('mesons')
        ninja = shutil.which('ninja')
        has_meson = meson is not None and ninja is not None
        build_dir = pyccel_dirpath / 'STC' / f'build-{compiler_family}'
        install_dir = pyccel_dirpath / 'STC' / 'install'
        if not build_dir.exists():
            if has_meson:
                buildtype = 'release'
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
                subprocess.run([make, f'BUILDDIR={build_dir}', '-C', self._src_dir],
                               check=True, cwd=pyccel_dirpath)
                shutil.copytree(ext_path / 'STC' / 'include', incdir)
                shutil.copyfile(build_dir / 'libstc.a', libdir / 'libstc.a')
                with open(libdir / 'pkgconfig' / 'stc.pc', 'w', encoding='utf-8') as f:
                    f.write("Name: stc\n")
                    f.write("Description: stc\n")
                    f.write("Version: 5.0-dev\n")
                    f.write(f"Libs: -L{libdir} -lstc -lm\n")
                    f.write(f"Cflags: -I{incdir}")

        current_PKG_CONFIG_PATH = os.environ.get('PKG_CONFIG_PATH', '')
        if current_PKG_CONFIG_PATH:
            current_PKG_CONFIG_PATH += ':'
        libdir = (install_dir / 'lib').glob(f'*-{compiler_family}')
        current_PKG_CONFIG_PATH += str(libdir / 'pkgconfig')
        os.environ['PKG_CONFIG_PATH'] = current_PKG_CONFIG_PATH

        assert self._check_for_package('stc')

#------------------------------------------------------------------------------------------

class GFTLCompileObj(ExternalCompileObj):
    def __init__(self):
        super().__init__("gFTL", src_dir = "GFTL-1.13")
        include = (self._src_dir / "include/v2/",)
        self._compile_obj = CompileObj("gFTL", folder = self._src_dir, has_target_file = False,
                                       include = include)

    def install_to(self, pyccel_dirpath):
        """
        Install the files to the Pyccel dirpath.

        Install the files to the Pyccel dirpath so they can be easily located and analysed by
        users. This function creates a symlink as the contents of this folder are not expected
        to change.

        Parameters
        ----------
        pyccel_dirpath : str | Path
            The path to the Pyccel working directory where the link should be created.
        """
        dest_dir = Path(pyccel_dirpath) / self._dest_dir
        if not dest_dir.exists():
            os.symlink(self._src_dir, dest_dir, target_is_directory=True)

#------------------------------------------------------------------------------------------

external_libs = {
        "stc" : STCCompileObj(),
        "gFTL" : GFTLCompileObj()
        }

internal_libs = {
    "pyc_math_f90"   : StdlibCompileObj("pyc_math_f90.f90", "math", libs = ('m',)),
    "pyc_math_c"     : StdlibCompileObj("pyc_math_c.c", "math"),
    "pyc_tools_f90"  : StdlibCompileObj("pyc_tools_f90.f90", "tools"),
    "cwrapper"       : CWrapperCompileObj("cwrapper.c", "cwrapper", accelerators=('python',)),
    "STC_Extensions" : StdlibCompileObj(".*.h", "STC_Extensions",
                                        has_target_file = False,
                                        dependencies = (external_libs['stc'].dependency,)),
    "gFTL_functions" : StdlibCompileObj("*.inc", "gFTL_functions",
                                        has_target_file = False,
                                        dependencies = (external_libs['gFTL'].dependency,))
}
