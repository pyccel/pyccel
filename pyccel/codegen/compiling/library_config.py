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
import pyccel.stdlib as stdlib_folder
import pyccel.extensions_install as ext_folder

#------------------------------------------------------------------------------------------

# get path to pyccel/stdlib/lib_name
stdlib_path = Path(stdlib_folder.__file__).parent

# get path to pyccel/extensions_install/lib_name
ext_path = Path(ext_folder.__file__).parent

#------------------------------------------------------------------------------------------

class StdlibCompileObj:
    def __init__(self, file_name, folder, **kwargs):
        self._src_dir = stdlib_path / folder
        self._file_name = file_name
        self._compile_obj_kwargs = {'folder' : folder, **kwargs}

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
        self._compile_obj = CompileObj(self._file_name, **self._compile_obj_kwargs)
        lib_dest_path = pyccel_dirpath / self._compile_obj_kwargs['folder']
        with FileLock(lib_dest_path + '.lock'):
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
                shutil.copytree(lib_path, lib_dest_path)

#------------------------------------------------------------------------------------------

class ExternalCompileObj:
    def __init__(self, dest_dir, src_dir = None, folder = None, include = (), libdir = (), **kwargs):
        src_dir = src_dir or dest_dir
        self._src_dir = ext_path / src_dir
        self._dest_dir = dest_dir
        folder = folder or src_dir
        include = tuple(self._src_dir / i for i in include)
        libdir = tuple(self._src_dir / i for i in libdir)
        self._compile_obj = CompileObj(folder = self._src_dir, **kwargs, has_target_file = False,
                                       include = include, libdir = libdir)

    @property
    def dependency(self):
        return self._compile_obj

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
        "stc" : ExternalCompileObj("STC", include = "include", libdir = "lib/.*", libs = "libstc.a"),
        "gFTL" : ExternalCompileObj("gFTL", src_dir = "GFTL-1.13", include = "include/v2/")
        }

internal_libs = {
    "pyc_math_f90"   : StdlibCompileObj("pyc_math_f90.f90", "math", libs = ('m',)),
    "pyc_math_c"     : StdlibCompileObj("pyc_math_c.c", "math"),
    "pyc_tools_f90"  : StdlibCompileObj("pyc_tools_f90.f90", "tools"),
    "cwrapper"       : StdlibCompileObj("cwrapper.c", "cwrapper", accelerators=('python',)),
    "STC_Extensions" : StdlibCompileObj(".*.h", "STC_Extensions",
                                        has_target_file = False,
                                        dependencies = (external_libs['stc'].dependency,)),
    "gFTL_functions" : StdlibCompileObj("*.inc", "gFTL_functions",
                                        has_target_file = False,
                                        dependencies = (external_libs['gFTL'].dependency,))
}
