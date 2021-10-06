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
import sysconfig
import platform
import warnings
from filelock import FileLock
from pyccel import __version__ as pyccel_version
from pyccel.errors.errors import Errors

errors = Errors()

if platform.system() == 'Darwin':
    # Set correct deployment target if on mac
    mac_target = platform.mac_ver()[0]
    if mac_target:
        os.environ['MACOSX_DEPLOYMENT_TARGET'] = mac_target

python_version = sysconfig.get_python_version()
def different_version(compiler):
    """
    Determine whether the specified compiler matches or differs from
    the expected version of pyccel and python
    """
    return compiler['pyccel_version'] != pyccel_version or \
            compiler['python_version'] != python_version

compilers_folder = os.path.join(os.path.dirname(__file__),'..','..','compilers')
with FileLock(compilers_folder+'.lock'):
    # TODO: Add an additional search location for user provided compiler files
    available_compilers = {f[:-5]:json.load(open(os.path.join(compilers_folder,f))) for f in os.listdir(compilers_folder)
                                                        if f.endswith('.json')}
    if len(available_compilers)==0 or \
            different_version(next(iter(available_compilers.values()))):
        from pyccel.compilers.generate_default import generate_default
        generate_default()
        available_compilers = {f[:-5]:json.load(open(os.path.join(compilers_folder,f))) for f in os.listdir(compilers_folder)
                                                            if f.endswith('.json')}

vendors = {c['family'] for c in available_compilers.values()}
sorted_compilers = {(c['family'],c['language']) : c for c in available_compilers.values()}

#------------------------------------------------------------
class Compiler:
    """
    Class which handles all compiler options

    Parameters
    ----------
    name  : str
               Name of the family of compilers
    language : str
               Language that we are translating to
    debug : bool
            Indicates whether we are compiling in debug mode
    """
    __slots__ = ('_debug','_info')
    def __init__(self, vendor : str, language : str, debug=False):
        if language=='python':
            return
        if vendor.endswith('.json') and os.path.exists(vendor):
            self._info = json.load(open(vendor))
            if language != self._info['language']:
                warnings.warn(UserWarning("Language does not match compiler. Using GNU compiler"))
                self._info = sorted_compilers[('GNU',language)]
        else:
            if vendor not in vendors:
                raise NotImplementedError("Unrecognised compiler vendor : {}".format(vendor))
            try:
                self._info = sorted_compilers[(vendor,language)]
            except KeyError as e:
                raise NotImplementedError("Compiler not available") from e

        self._debug = debug

    def _get_exec(self, accelerators):
        # Get executable
        exec_cmd = self._info['mpi_exec'] if 'mpi' in accelerators else self._info['exec']

        if shutil.which(exec_cmd) is None:
            errors.report("Could not find compiler ({})".format(exec_cmd),
                    severity='fatal')

        return exec_cmd

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
        Provide all components required for compiling

        Parameters
        ----------
        compile_obj  : CompileObj
                       Object containing all information about the object to be compiled
        accelerators : iterable of str
                       Name of all tools used by the code which require additional flags/includes/etc

        Results
        -------
        exec_cmd      : str
                        The command required to run the executable
        inc_flags     : iterable of strs
                        The include directories required to compile
        libs_flags    : iterable of strs
                        The libraries required to compile
        libdirs_flags : iterable of strs
                        The directories containing libraries required to compile
        m_code        : iterable of strs
                        The objects required to compile
        """

        # get includes
        includes = self._get_includes(compile_obj.includes, accelerators)
        inc_flags = self._insert_prefix_to_list(includes, '-I')

        # Get dependencies (.o/.a)
        m_code = self._get_dependencies(compile_obj.extra_modules, accelerators)

        # Get libraries and library directories
        libs = self._get_libs(compile_obj.libs, accelerators)
        libs_flags = [s if s.startswith('-l') else '-l{}'.format(s) for s in libs]
        libdirs = self._get_libdirs(compile_obj.libdirs, accelerators)
        libdirs_flags = self._insert_prefix_to_list(libdirs, '-L')

        exec_cmd = self._get_exec(accelerators)

        return exec_cmd, inc_flags, libs_flags, libdirs_flags, m_code

    def compile_module(self, compile_obj, output_folder, verbose = False):
        """
        Compile a module

        Parameters
        ----------
        compile_obj   : CompileObj
                        Object containing all information about the object to be compiled
        output_folder : str
                        The folder where the result should be saved
        verbose       : bool
                        Indicates whether additional output should be shown
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

        with FileLock('.lock_acquisition.lock'):
            compile_obj.acquire_lock()
        try:
            self.run_command(cmd, verbose)
        finally:
            compile_obj.release_lock()

    def compile_program(self, compile_obj, output_folder, verbose = False):
        """
        Compile a program

        Parameters
        ----------
        compile_obj   : CompileObj
                        Object containing all information about the object to be compiled
        output_folder : str
                        The folder where the result should be saved
        verbose       : bool
                        Indicates whether additional output should be shown
        """
        accelerators = compile_obj.accelerators

        # get flags
        flags = self._get_flags(compile_obj.flags, accelerators)

        # Get compile options
        exec_cmd, includes, libs_flags, libdirs_flags, m_code = \
                self._get_compile_components(compile_obj, accelerators)

        if self._info['language'] == 'fortran':
            j_code = (self._info['module_output_flag'], output_folder)
        else:
            j_code = ()

        cmd = [exec_cmd, *flags, *includes, *libdirs_flags,
                *m_code, compile_obj.source,
                '-o', compile_obj.program_target,
                *libs_flags, *j_code]

        with FileLock('.lock_acquisition.lock'):
            compile_obj.acquire_lock()
        try:
            self.run_command(cmd, verbose)
        finally:
            compile_obj.release_lock()

        return compile_obj.program_target

    def compile_shared_library(self, compile_obj, output_folder, verbose = False, sharedlib_modname=None):
        """
        Compile a module to a shared library

        Parameters
        ----------
        compile_obj   : CompileObj
                        Object containing all information about the object to be compiled
        output_folder : str
                        The folder where the result should be saved
        verbose       : bool
                        Indicates whether additional output should be shown

        Returns
        -------
        file_out : str
                   Generated library name
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
                *m_code, compile_obj.module_target,
                '-o', file_out, *libs_flags]

        with FileLock('.lock_acquisition.lock'):
            compile_obj.acquire_lock()
        try:
            self.run_command(cmd, verbose)
        finally:
            compile_obj.release_lock()

        return file_out

    @staticmethod
    def run_command(cmd, verbose):
        """
        Run the provided command and collect the output

        Parameters
        ----------
        cmd     : iterable
                  The command to run
        verbose : bool
                  Indicates whether additional output should be shown
        """
        cmd = [os.path.expandvars(c) for c in cmd]
        if verbose:
            print(' '.join(cmd))

        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True)
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
