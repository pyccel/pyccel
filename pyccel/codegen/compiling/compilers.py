#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
Module handling everything related to the compilers used to compile the various generated files
"""
import json
import os
import shutil
import subprocess
import platform
import warnings
from pyccel.compilers.default_compilers import available_compilers, vendors
from pyccel.errors.errors import Errors

errors = Errors()

if platform.system() == 'Darwin':
    # Collect version using mac tools to avoid unexpected results on Big Sur
    # https://developer.apple.com/documentation/macos-release-notes/macos-big-sur-11_0_1-release-notes#Third-Party-Apps
    with subprocess.Popen([shutil.which("sw_vers"), "-productVersion"], stdout=subprocess.PIPE) as p:
        result, err = p.communicate()
    mac_version_tuple = result.decode("utf-8").strip().split('.')
    mac_target = '.'.join(mac_version_tuple[:2])
    os.environ['MACOSX_DEPLOYMENT_TARGET'] = mac_target


def get_condaless_search_path(conda_warnings = 'basic'):
    """
    Get a list of paths excluding the conda paths.

    Get the value of the PATH variable to be set when searching for the compiler
    This is the same as the environment PATH variable but without any conda paths.

    Parameters
    ----------
    conda_warnings : str, optional
        Specify the level of Conda warnings to display (choices: off, basic, verbose), Default is 'basic'.

    Returns
    -------
    str
        A list of paths excluding the conda paths.
    """
    path_sep = ';' if platform.system() == 'Windows' else ':'
    current_path = os.environ['PATH']
    folders = {f: f.split(os.sep) for f in current_path.split(path_sep)}
    conda_folder_names = ('conda', 'anaconda', 'miniconda',
                          'Conda', 'Anaconda', 'Miniconda')
    conda_folders = [p for p,f in folders.items() if any(con in f for con in conda_folder_names)]
    if conda_folders:
        if conda_warnings in ('basic', 'verbose'):
            message_warning = "Conda paths are ignored. See https://github.com/pyccel/pyccel/blob/devel/tutorial/compiler.md#utilising-pyccel-within-anaconda-environment for details"
            if conda_warnings == 'verbose':
                message_warning = message_warning + "\nConda ignored PATH:\n"
                message_warning = message_warning + ":".join(conda_folders)
            warnings.warn(UserWarning(message_warning))
    acceptable_search_paths = path_sep.join(p for p in folders.keys() if p not in conda_folders and os.path.exists(p))
    return acceptable_search_paths

#------------------------------------------------------------
class Compiler:
    """
    Class which handles all compiler options.

    This class uses the compiler vendor or a json file to collect
    all compiler configuration parameters. These are then used to
    correctly print compiler commands such as shared library
    compilation commands or executable creation commands.

    Parameters
    ----------
    vendor : str
               Name of the family of compilers.
    language : str
               Language that we are translating to.
    debug : bool
               Indicates whether we are compiling in debug mode.
    """
    __slots__ = ('_debug','_info')
    acceptable_bin_paths = None
    def __init__(self, vendor : str, language : str, debug=False):
        if language=='python':
            return
        if vendor.endswith('.json') and os.path.exists(vendor):
            with open(vendor, encoding="utf-8") as vendor_file:
                self._info = json.load(vendor_file)
            if language != self._info['language']:
                warnings.warn(UserWarning("Language does not match compiler. Using GNU compiler"))
                self._info = available_compilers[('GNU',language)]
        else:
            if vendor not in vendors:
                raise NotImplementedError(f"Unrecognised compiler vendor : {vendor}")
            try:
                self._info = available_compilers[(vendor,language)]
            except KeyError as e:
                raise NotImplementedError("Compiler not available") from e

        self._debug = debug

    def _get_exec(self, accelerators):
        """
        Obtain the path of the executable based on the specified accelerators.

        The `_get_exec` method is responsible for retrieving the path of the executable based on the specified accelerators.
        It is used internally in the Pyccel module.

        Parameters
        ----------
        accelerators : str
            Specifies the accelerators to be used.

        Returns
        -------
        str
            The path of the executable corresponding to the specified accelerators.

        Raises
        ------
        PyccelError
            If the compiler executable cannot be found.
        """
        # Get executable
        exec_cmd = self._info['mpi_exec'] if 'mpi' in accelerators else self._info['exec']

        # Clean conda paths out of the PATH variable
        current_path = os.environ['PATH']
        os.environ['PATH'] = self.acceptable_bin_paths

        # Find the exact path of the executable
        exec_loc = shutil.which(exec_cmd)

        # Reset PATH variable
        os.environ['PATH'] = current_path

        if exec_loc is None:
            errors.report(f"Could not find compiler ({exec_cmd})",
                    severity='fatal')

        return exec_loc

    def _get_flags(self, flags = (), accelerators = ()):
        """
        Collect necessary compile flags

        Parameters
        ----------
        flags        : iterable of str
                       Any additional flags requested by the user
                       / required by the file
        accelerators : iterable or str
                       Accelerators used by the code
        """
        flags = list(flags)

        if self._debug:
            flags.extend(self._info.get('debug_flags',()))
        else:
            flags.extend(self._info.get('release_flags',()))

        flags.extend(self._info.get('general_flags',()))
        # M_PI is not in the standard
        #if 'python' not in accelerators:
        #    # Python sets its own standard
        #    flags.extend(self._info.get('standard_flags',()))

        for a in accelerators:
            flags.extend(self._info.get(a,{}).get('flags',()))

        return flags

    def _get_property(self, key, prop = (), accelerators = ()):
        """
        Collect necessary compile property

        Parameters
        ----------
        property     : iterable of str
                       Any additional values of the property
                       requested by the user / required by the file
        accelerators : iterable or str
                       Accelerators used by the code
        """
        # Use dict keys as an ordered set
        prop = dict.fromkeys(prop)

        prop.update(dict.fromkeys(self._info.get(key,())))

        for a in accelerators:
            prop.update(dict.fromkeys(self._info.get(a,{}).get(key,())))

        return prop.keys()

    def _get_includes(self, includes = (), accelerators = ()):
        """
        Collect necessary compile include directories

        Parameters
        ----------
        includes     : iterable of str
                       Any additional include directories requested by the user
                       / required by the file
        accelerators : iterable or str
                       Accelerators used by the code
        """
        return self._get_property('includes', includes, accelerators)

    def _get_libs(self, libs = (), accelerators = ()):
        """
        Collect necessary compile libraries

        Parameters
        ----------
        libs         : iterable of str
                       Any additional libraries requested by the user
                       / required by the file
        accelerators : iterable or str
                       Accelerators used by the code
        """
        return self._get_property('libs', libs, accelerators)

    def _get_libdirs(self, libdirs = (), accelerators = ()):
        """
        Collect necessary compile library directories

        Parameters
        ----------
        libdirs      : iterable of str
                       Any additional library directories
                       requested by the user / required by the file
        accelerators : iterable or str
                       Accelerators used by the code
        """
        return self._get_property('libdirs', libdirs, accelerators)

    def _get_dependencies(self, dependencies = (), accelerators = ()):
        """
        Collect necessary dependencies

        Parameters
        ----------
        dependencies : iterable of str
                       Any additional dependencies required by the file
        accelerators : iterable or str
                       Accelerators used by the code
        """
        return self._get_property('dependencies', dependencies, accelerators)

    @staticmethod
    def _insert_prefix_to_list(lst, prefix):
        """
        Add a prefix into a list. E.g:
        >>> lst = [1, 2, 3]
        >>> _insert_prefix_to_list(lst, 'num:')
        ['num:', 1, 'num:', 2, 'num:', 3]

        Parameters
        ----------
        lst    : iterable
                 The list into which the prefix is inserted
        prefix : str
                 The prefix
        """
        lst = [(prefix, i) for i in lst]
        return [f for fi in lst for f in fi]

    def _get_compile_components(self, compile_obj, accelerators = ()):
        """
        Provide all components required for compiling.

        Provide all the different componenets (include directories, libraries, etc)
        which are needed in order to compile any file.

        Parameters
        ----------
        compile_obj : CompileObj
            Object containing all information about the object to be compiled.
        accelerators : iterable of str
            Name of all tools used by the code which require additional flags/includes/etc.

        Returns
        -------
        exec_cmd : str
            The command required to run the executable.
        inc_flags : iterable of strs
            The include directories required to compile.
        libs_flags : iterable of strs
            The libraries required to compile.
        libdirs_flags : iterable of strs
            The directories containing libraries required to compile.
        m_code : iterable of strs
            The objects required to compile.
        """

        # get includes
        includes = self._get_includes(compile_obj.includes, accelerators)
        inc_flags = self._insert_prefix_to_list(includes, '-I')

        # Get dependencies (.o/.a)
        m_code = self._get_dependencies(compile_obj.extra_modules, accelerators)

        # Get libraries and library directories
        libs = self._get_libs(compile_obj.libs, accelerators)
        libs_flags = [s if s.startswith('-l') else f'-l{s}' for s in libs]
        libdirs = self._get_libdirs(compile_obj.libdirs, accelerators)
        libdirs_flags = self._insert_prefix_to_list(libdirs, '-L')

        exec_cmd = self._get_exec(accelerators)

        return exec_cmd, inc_flags, libs_flags, libdirs_flags, m_code

    def compile_module(self, compile_obj, output_folder, verbose = False):
        """
        Compile a module.

        Compile a file containing a module to a .o file.

        Parameters
        ----------
        compile_obj : CompileObj
            Object containing all information about the object to be compiled.

        output_folder : str
            The folder where the result should be saved.

        verbose : bool
            Indicates whether additional output should be shown.
        """
        accelerators = compile_obj.accelerators

        # Get flags
        flags = self._get_flags(compile_obj.flags, accelerators)
        flags.append('-c')

        # Get includes
        includes  = self._get_includes(compile_obj.includes, accelerators)
        inc_flags = self._insert_prefix_to_list(includes, '-I')

        # Get executable
        exec_cmd = self._get_exec(accelerators)

        if self._info['language'] == 'fortran':
            j_code = (self._info['module_output_flag'], output_folder)
        else:
            j_code = ()

        cmd = [exec_cmd, *flags, *inc_flags,
                compile_obj.source, '-o', compile_obj.module_target,
                *j_code]

        with compile_obj:
            self.run_command(cmd, verbose)

    def compile_program(self, compile_obj, output_folder, verbose = False):
        """
        Compile a program.

        Compile a file containing a program to an executable.

        Parameters
        ----------
        compile_obj : CompileObj
            Object containing all information about the object to be compiled.

        output_folder : str
            The folder where the result should be saved.

        verbose : bool
            Indicates whether additional output should be shown.

        Returns
        -------
        str
            The name of the generated executable.
        """
        accelerators = compile_obj.accelerators

        # get flags
        flags = self._get_flags(compile_obj.flags, accelerators)

        # Get compile options
        exec_cmd, includes, libs_flags, libdirs_flags, m_code = \
                self._get_compile_components(compile_obj, accelerators)
        linker_libdirs_flags = ['-Wl,-rpath' if l == '-L' else l for l in libdirs_flags]

        if self._info['language'] == 'fortran':
            j_code = (self._info['module_output_flag'], output_folder)
        else:
            j_code = ()

        cmd = [exec_cmd, *flags, *includes, *libdirs_flags,
                 *linker_libdirs_flags, *m_code, compile_obj.source,
                '-o', compile_obj.program_target,
                *libs_flags, *j_code]

        with compile_obj:
            self.run_command(cmd, verbose)

        return compile_obj.program_target

    def compile_shared_library(self, compile_obj, output_folder, verbose = False, sharedlib_modname=None):
        """
        Compile a module to a shared library.

        Compile a file containing a module with C-API calls to a shared library which can
        be called from Python.

        Parameters
        ----------
        compile_obj : CompileObj
            Object containing all information about the object to be compiled.

        output_folder : str
            The folder where the result should be saved.

        verbose : bool
            Indicates whether additional output should be shown.

        sharedlib_modname : str, optional
            The name of the library that should be generated. If none is provided then it
            defaults to matching the name of the file.

        Returns
        -------
        str
            Generated library name.
        """
        # Ensure python options are collected
        accelerators = set(compile_obj.accelerators)

        accelerators.remove('python')

        # get flags
        flags = self._get_flags(compile_obj.flags, accelerators)

        accelerators.add('python')

        # Collect compile information
        exec_cmd, includes, libs_flags, libdirs_flags, m_code = \
                self._get_compile_components(compile_obj, accelerators)
        linker_libdirs_flags = ['-Wl,-rpath' if l == '-L' else l for l in libdirs_flags]

        flags.insert(0,"-shared")

        # Get name of file
        ext_suffix = self._info['python']['shared_suffix']
        sharedlib_modname = sharedlib_modname or compile_obj.python_module
        file_out = os.path.join(compile_obj.source_folder, sharedlib_modname+ext_suffix)

        cmd = [exec_cmd, *flags, *includes, *libdirs_flags, *linker_libdirs_flags,
                compile_obj.module_target, *m_code,
                '-o', file_out, *libs_flags]

        with compile_obj:
            self.run_command(cmd, verbose)

        return file_out

    @staticmethod
    def run_command(cmd, verbose):
        """
        Run the provided command and collect the output.

        Run the provided compilation command, collect the output and raise any
        necessary errors if the file does not compile.

        Parameters
        ----------
        cmd : list of str
            The command to run.
        verbose : bool
            Indicates whether additional output should be shown.

        Returns
        -------
        str
            The exact command that was run.

        Raises
        ------
        RuntimeError
            Raises `RuntimeError` if the file does not compile.
        """
        cmd = [os.path.expandvars(c) for c in cmd]
        if verbose:
            print(' '.join(cmd))

        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True) as p:
            out, err = p.communicate()

        if verbose and out:
            print(out)
        if p.returncode != 0:
            err_msg = "Failed to build module"
            err_msg += "\n" + err
            raise RuntimeError(err_msg)
        if err:
            warnings.warn(UserWarning(err))

        return cmd

    def export_compiler_info(self, compiler_export_file):
        """
        Export the compiler configuration to a json file.

        Print the information describing all compiler options to the
        specified file in json format. This file can be used for
        debugging purposes or it can be manually modified and fed
        back to Pyccel to correct compilation problems or request
        more unusual flags/include directories/etc.

        Parameters
        ----------
        compiler_export_file : str
            The name of the file where the compiler configuration
            should be printed.
        """
        with open(compiler_export_file,'w', encoding="utf-8") as out_file:
            print(json.dumps(self._info, indent=4),
                    file=out_file)
