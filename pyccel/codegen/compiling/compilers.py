#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
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
            message_warning = "Conda paths are ignored. See https://github.com/pyccel/pyccel/blob/devel/docs/compiler.md#utilising-pyccel-within-anaconda-environment for details"
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
    debug : bool
               Indicates whether we are compiling in debug mode.
    """
    __slots__ = ('_debug','_compiler_info','_language_info')
    acceptable_bin_paths = None
    def __init__(self, vendor : str, debug=False):
        if vendor.endswith('.json') and os.path.exists(vendor):
            with open(vendor, encoding="utf-8") as vendor_file:
                self._compiler_info = json.load(vendor_file)
        else:
            if vendor not in vendors:
                raise NotImplementedError(f"Unrecognised compiler vendor : {vendor}")
            try:
                self._compiler_info = available_compilers[vendor]
            except KeyError as e:
                raise NotImplementedError("Compiler not available") from e

        self._debug = debug
        self._language_info = None

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
        exec_cmd = self._language_info['mpi_exec'] if 'mpi' in accelerators else self._language_info['exec']

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
        Collect necessary compile flags.

        Collect necessary compile flags, e.g. those relevant to the
        language or compilation mode (debug/release).

        Parameters
        ----------
        flags : iterable of str
            Any additional flags requested by the user / required by
            the file.
        accelerators : iterable or str
            Accelerators used by the code.

        Returns
        -------
        list[str]
            A list containing the flags.
        """
        flags = list(flags)

        if self._debug:
            flags.extend(self._language_info.get('debug_flags',()))
        else:
            flags.extend(self._language_info.get('release_flags',()))

        flags.extend(self._language_info.get('general_flags',()))
        # M_PI is not in the standard
        #if 'python' not in accelerators:
        #    # Python sets its own standard
        #    flags.extend(self._language_info.get('standard_flags',()))

        for a in accelerators:
            flags.extend(self._language_info.get(a,{}).get('flags',()))

        return flags

    def _get_property(self, key, properties = (), accelerators = ()):
        """
        Collect necessary compile property.

        Collect necessary compile properties such as include folders
        or library directories.

        Parameters
        ----------
        key : str
            A key describing the property of interest.
        properties : iterable of str
            Any additional values of the property requested by the
            user / required by the file.
        accelerators : iterable or str
            Accelerators used by the code.

        Returns
        -------
        iterable[str]
            An iterable containing the relevant information from the
            requested property.

        Examples
        --------
        >> self._get_property("libs", ("-lmy_lib",), ())
        dict_keys(['-lmy_lib', '-lm'])

        >> self._get_property("libs", ("-lmy_lib",), ("openmp",))
        dict_keys(['-lmy_lib', '-lm', 'gomp'])

        >> self._get_property("include", ("/home/user/homemade-install-dir/",), ("mpi",))
        dict_keys(['/home/user/homemade-install-dir/'])
        """
        # Use a dictionary instead of a set to ensure properties are ordered by insertion
        # The keys of the dictionary contain the values for the property of interest.
        properties = dict.fromkeys(properties)

        properties.update(dict.fromkeys(self._language_info.get(key,())))

        for a in accelerators:
            properties.update(dict.fromkeys(self._language_info.get(a,{}).get(key,())))

        return properties.keys()

    def _get_include(self, include = (), accelerators = ()):
        """
        Collect necessary compile include directories.

        Collect necessary compile include directories.

        Parameters
        ----------
        include : iterable of str
                       Any additional include directories requested by the user
                       / required by the file.
        accelerators : iterable or str
                       Accelerators used by the code.

        Returns
        -------
        list[str]
            A list of the include folders.
        """
        return self._get_property('include', include, accelerators)

    def _get_libs(self, libs = (), accelerators = ()):
        """
        Collect necessary compile libraries.

        Collect necessary compile libraries.

        Parameters
        ----------
        libs : iterable of str
            Any additional libraries requested by the user / required
            by the file.
        accelerators : iterable or str
            Accelerators used by the code.

        Returns
        -------
        list[str]
            A list of the libraries.
        """
        return self._get_property('libs', libs, accelerators)

    def _get_libdir(self, libdir = (), accelerators = ()):
        """
        Collect necessary compile library directories.

        Collect necessary compile library directories.

        Parameters
        ----------
        libdir : iterable of str
            Any additional library directories requested by the user
            / required by the file.
        accelerators : iterable or str
            Accelerators used by the code.

        Returns
        -------
        list[str]
            A list of the folders containing libraries.
        """
        return self._get_property('libdir', libdir, accelerators)

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
        Add a prefix into a list.

        Add a prefix into a list. E.g:
        >>> lst = [1, 2, 3]
        >>> _insert_prefix_to_list(lst, 'num:')
        ['num:', 1, 'num:', 2, 'num:', 3]

        Parameters
        ----------
        lst : iterable
            This sequence is copied to a new list with `prefix` before each element.
        prefix : Any
            The prefix to be placed before each element of `lst`.

        Returns
        -------
        list
            The list with the prefix inserted.
        """
        lst = [(prefix, i) for i in lst]
        return [f for fi in lst for f in fi]

    def _get_compile_components(self, compile_obj, accelerators = ()):
        """
        Provide all components required for compiling.

        Provide all the different components (include directories, libraries, etc)
        which are needed in order to compile any file.

        Parameters
        ----------
        compile_obj : CompileObj
            Object containing all information about the object to be compiled.
        accelerators : iterable of str
            Name of all tools used by the code which require additional flags/include/etc.

        Returns
        -------
        exec_cmd : str
            The command required to run the executable.
        inc_flags : iterable of strs
            The include directories required to compile.
        libs_flags : iterable of strs
            The libraries required to compile.
        libdir_flags : iterable of strs
            The directories containing libraries required to compile.
        m_code : iterable of strs
            The objects required to compile.
        """

        # get include
        include = self._get_include(compile_obj.include, accelerators)
        inc_flags = self._insert_prefix_to_list(include, '-I')

        # Get dependencies (.o/.a)
        m_code = self._get_dependencies(compile_obj.extra_modules, accelerators)

        # Get libraries and library directories
        libs = self._get_libs(compile_obj.libs, accelerators)
        libs_flags = [s if s.startswith('-l') else f'-l{s}' for s in libs]
        libdir = self._get_libdir(compile_obj.libdir, accelerators)
        libdir_flags = self._insert_prefix_to_list(libdir, '-L')

        exec_cmd = self._get_exec(accelerators)

        return exec_cmd, inc_flags, libs_flags, libdir_flags, m_code

    def compile_module(self, compile_obj, output_folder, language, verbose):
        """
        Compile a module.

        Compile a file containing a module to a .o file.

        Parameters
        ----------
        compile_obj : CompileObj
            Object containing all information about the object to be compiled.

        output_folder : str
            The folder where the result should be saved.

        language : str
            Language that we are compiling.

        verbose : int
            Indicates the level of verbosity.
        """
        if not compile_obj.has_target_file:
            return

        if verbose:
            print(">> Compiling :: ", compile_obj.module_target)

        self._language_info = self._compiler_info[language]

        accelerators = compile_obj.accelerators

        # Get flags
        flags = self._get_flags(compile_obj.flags, accelerators)
        flags.append('-c')

        # Get include
        include  = self._get_include(compile_obj.include, accelerators)
        inc_flags = self._insert_prefix_to_list(include, '-I')

        # Get executable
        exec_cmd = self._get_exec(accelerators)

        if language == 'fortran':
            j_code = (self._language_info['module_output_flag'], output_folder)
        else:
            j_code = ()

        cmd = [exec_cmd, *flags, *inc_flags,
                compile_obj.source, '-o', compile_obj.module_target,
                *j_code]

        with compile_obj:
            self.run_command(cmd, verbose)

        self._language_info = None

    def compile_program(self, compile_obj, output_folder, language, verbose):
        """
        Compile a program.

        Compile a file containing a program to an executable.

        Parameters
        ----------
        compile_obj : CompileObj
            Object containing all information about the object to be compiled.

        output_folder : str
            The folder where the result should be saved.

        language : str
            Language that we are compiling.

        verbose : int
            Indicates the level of verbosity.

        Returns
        -------
        str
            The name of the generated executable.
        """
        if verbose:
            print(">> Compiling executable :: ", compile_obj.program_target)

        self._language_info = self._compiler_info[language]

        accelerators = compile_obj.accelerators

        # get flags
        flags = self._get_flags(compile_obj.flags, accelerators)

        # Get compile options
        exec_cmd, include, libs_flags, libdir_flags, m_code = \
                self._get_compile_components(compile_obj, accelerators)
        linker_libdir_flags = ['-Wl,-rpath' if l == '-L' else l for l in libdir_flags]

        if language == 'fortran':
            j_code = (self._language_info['module_output_flag'], output_folder)
        else:
            j_code = ()

        cmd = [exec_cmd, *flags, *include, *libdir_flags,
                 *linker_libdir_flags, *m_code, compile_obj.source,
                '-o', compile_obj.program_target,
                *libs_flags, *j_code]

        with compile_obj:
            self.run_command(cmd, verbose)

        self._language_info = None

        return compile_obj.program_target

    def compile_shared_library(self, compile_obj, output_folder, language, verbose, sharedlib_modname=None):
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

        language : str
            Language that we are compiling.

        verbose : int
            Indicates the level of verbosity.

        sharedlib_modname : str, optional
            The name of the library that should be generated. If none is provided then it
            defaults to matching the name of the file.

        Returns
        -------
        str
            Generated library name.
        """
        self._language_info = self._compiler_info[language]

        # Ensure python options are collected
        accelerators = set(compile_obj.accelerators)

        accelerators.remove('python')

        # get flags
        flags = self._get_flags(compile_obj.flags, accelerators)

        accelerators.add('python')

        # Collect compile information
        exec_cmd, _, libs_flags, libdir_flags, m_code = \
                self._get_compile_components(compile_obj, accelerators)
        linker_libdir_flags = ['-Wl,-rpath' if l == '-L' else l for l in libdir_flags]

        flags.insert(0,"-shared")

        # Get name of file
        ext_suffix = self._language_info['python']['shared_suffix']
        sharedlib_modname = sharedlib_modname or compile_obj.python_module
        file_out = os.path.join(compile_obj.source_folder, sharedlib_modname+ext_suffix)

        if verbose:
            print(">> Compiling shared library :: ", file_out)

        cmd = [exec_cmd, *flags, *libdir_flags, *linker_libdir_flags,
                compile_obj.module_target, *m_code,
                '-o', file_out, *libs_flags]

        with compile_obj:
            self.run_command(cmd, verbose)

        self._language_info = None

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
        verbose : int
            Indicates the level of verbosity.

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
        if verbose > 1:
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
            print(json.dumps(self._compiler_info, indent=4),
                    file=out_file)
