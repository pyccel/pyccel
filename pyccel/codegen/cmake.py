# coding: utf-8

import os

# ... TODO: - check that cmake exists in PATH
#           - create build dir for cmake
#           - add method make
#           - add method install
#           - add logger

class CMake(object):
    """User-friendly class for cmake."""
    def __init__(self, path, \
                 prefix, \
                 flags, \
                 flags_fortran, \
                 compiler_fortran, files=None):
        """
        Constructor for cmake.

        path: str
            path to CMakeLists of the current project

        prefix: str
            installation directory

        flags: dict
            general flags from the config file

        flags_fortran: dict
            flags for the fortran compiler

        compiler_fortran: str
            a valid fortran compiler

        files: list
            list of files to compile
        """
        # ...
        self._flags            = flags
        self._flags_fortran    = flags_fortran
        self._compiler_fortran = compiler_fortran
        self._prefix           = prefix
        self._path             = path
        self._files            = files
        # ...

        # ...
        if files:
            self.initialize(files)
        # ...

        # ... create build dir
        build_path = os.path.join(path, 'build')
        if not os.path.exists(build_path):
            os.makedirs(build_path)
        self._build_path = build_path
        # ...

        # ...
        args = []
        # ...

        # ... MPI
        if "mpi" in compiler_fortran.lower():
            args += ['-DMPI_ENABLED=ON']
        # ...

        # ...
        args += [' -DCMAKE_INSTALL_PREFIX={0}'.format(prefix)]
        args += [' -DCMAKE_Fortran_FLAGS="{0}"'.format(self.flags_fortran)]
#        args += [' -DCMAKE_Fortran_FLAGS_DEBUG={0}'.format(self.flags_fortran)]
#        args += [' -DCMAKE_Fortran_FLAGS_RELEASE={0}'.format(self.flags_fortran)]
        # ...

        # ...
        for flag, value in list(flags.items()):
            args += ['-D{0}={1}'.format(flag, value)]
        # ...

        # ...
        self._args = args
        # ...

    @property
    def args(self):
        return self._args

    @property
    def flags(self):
        return self._flags

    @property
    def flags_fortran(self):
        return self._flags_fortran

    @property
    def compiler_fortran(self):
        return self._compiler_fortran

    @property
    def prefix(self):
        return self._prefix

    @property
    def path(self):
        return self._path

    @property
    def build_path(self):
        return self._build_path

    @property
    def files(self):
        return self._files

    def configure(self, verbose=True):
        # ...
        options = ' '.join(i for i in self.args)
        cmd = 'cmake {0} ..'.format(options)
        # ...

        # ...
        base_dir = os.getcwd()

        os.chdir(self.build_path)
        os.system(cmd)

        if verbose:
            f = open('clapp-manager.log', 'w')
            f.write('# this file was generated automatically by clappmanager.\n')
            f.write('# cmake configuration using the command line: \n')
            f.write('\n')
            f.write(cmd)
            f.close()

        os.chdir(base_dir)
        # ...

    def make(self, verbose=False):
        # ...
        cmd = 'make'
        if verbose:
            cmd = '{0} VERBOSE=1'.format(cmd)
        # ...

        # ...
        base_dir = os.getcwd()

        os.chdir(self.build_path)
        os.system(cmd)
        os.chdir(base_dir)
        # ...

    def install(self):
        # ...
        cmd = 'make install'
        # ...

        # ...
        base_dir = os.getcwd()

        os.chdir(self.build_path)
        os.system(cmd)
        os.chdir(base_dir)
        # ...

    def initialize(self, files):
        pass

