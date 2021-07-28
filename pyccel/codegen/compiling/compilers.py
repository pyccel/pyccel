"""
Module handling everything related to the compilers used to compile the various generated files
"""
import json
import os
import subprocess
import sysconfig
import warnings
from pyccel import __version__ as pyccel_version

# Set correct deployment target if on mac
mac_target = sysconfig.get_config_var('MACOSX_DEPLOYMENT_TARGET')
if mac_target:
    os.environ['MACOSX_DEPLOYMENT_TARGET'] = mac_target

compilers_folder = os.path.join(os.path.dirname(__file__),'..','..','compilers')
available_compilers = {f[:-5]:json.load(open(os.path.join(compilers_folder,f))) for f in os.listdir(compilers_folder)
                                                    if f.endswith('.json')}
if len(available_compilers)==0 or \
        next(iter(available_compilers.values()))['pyccel_version'] != pyccel_version:
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
    def __init__(self, vendor : str, language : str, debug=False):
        if vendor not in vendors:
            raise NotImplementedError("Unrecognised compiler vendor : {}".format(vendor))
        try:
            self._info = sorted_compilers[(vendor,language)]
        except KeyError as e:
            raise NotImplementedError("Compiler not available") from e

        self._debug = debug

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

        flags.extend(self._info.get('standard_flags',()))

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
        prop = list(prop)

        prop.extend(self._info.get(key,()))

        for a in accelerators:
            prop.extend(self._info.get(a,{}).get(key,()))

        return prop

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

    def _get_compile_componenets(self, compile_obj, accelerators = ()):
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
        flags         : iterable of strs
                        The flags required to compile
        inc_flags     : iterable of strs
                        The include directories required to compile
        libs_flags    : iterable of strs
                        The libraries required to compile
        libdirs_flags : iterable of strs
                        The directories containing libraries required to compile
        m_code        : iterable of strs
                        The objects required to compile
        """
        # get flags
        flags = self._get_flags(compile_obj.flags, accelerators)

        # get includes
        includes = self._get_includes(compile_obj.includes, accelerators)
        inc_flags = self._insert_prefix_to_list(includes, '-I')

        # Get dependencies (.o/.a)
        m_code = self._get_dependencies(compile_obj.extra_modules, accelerators)

        # Get libraries and library directories
        libs = self._get_libs(compile_obj.libs, accelerators)
        libs_flags = ['-l{}'.format(s) for s in libs]
        libdirs = self._get_libdirs(compile_obj.libdirs, accelerators)
        libdirs_flags = self._insert_prefix_to_list(libdirs, '-L')

        # Get executable
        exec_cmd = self._info['exec'] if 'mpi' in accelerators else self._info['exec']

        return exec_cmd, flags, inc_flags, libs_flags, libdirs_flags, m_code

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
        exec_cmd = self._info['exec'] if 'mpi' in accelerators else self._info['exec']

        if self._info['language'] == 'fortran':
            j_code = (self._info['module_output_flag'], output_folder)
        else:
            j_code = ()

        cmd = [exec_cmd, *flags, *inc_flags,
                compile_obj.source, '-o', compile_obj.target,
                *j_code]

        compile_obj.acquire_lock()
        self.run_command(cmd, verbose)
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

        # Get compile options
        exec_cmd, flags, includes, libs_flags, libdirs_flags, m_code = \
                self._get_compile_componenets(compile_obj, verbose, accelerators)

        if self._info['language'] == 'fortran':
            j_code = (self._info['module_output_flag'], output_folder)
        else:
            j_code = ()

        if compile_obj.is_module:
            flags.append('-c')

        cmd = [exec_cmd, *flags, *includes, *libdirs_flags,
                *m_code, compile_obj.source,
                '-o', compile_obj.target,
                *libs_flags, *j_code]

        compile_obj.acquire_lock()
        self.run_command(cmd, verbose)
        compile_obj.release_lock()

    def compile_file(self, compile_obj, output_folder, verbose = False):
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
        if compile_obj.is_module:
            self.compile_module(compile_obj, output_folder, verbose)
        else:
            self.compile_program(compile_obj, output_folder, verbose)

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
        accelerators = list(compile_obj.accelerators)
        if 'python' not in accelerators:
            accelerators.append('python')

        # Collect compile information
        exec_cmd, flags, includes, libs_flags, libdirs_flags, m_code = \
                self._get_compile_componenets(compile_obj, accelerators)

        # Include linker flags to generate shared object
        flags.extend(self._info['python']['linker_flags'])

        # Get name of file
        ext_suffix = self._info['python']['shared_suffix']
        sharedlib_modname = sharedlib_modname or compile_obj.python_module
        file_out = os.path.join(compile_obj.source_folder, sharedlib_modname+ext_suffix)

        cmd = [exec_cmd, *flags, *includes, *libdirs_flags,
                *libs_flags, *m_code, compile_obj.target,
                '-o', file_out]

        compile_obj.acquire_lock()
        self.run_command(cmd, verbose)
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
        if verbose:
            print(' '.join(cmd))

        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
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
