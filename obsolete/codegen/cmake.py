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
                 prefix=None, \
                 flags=None, \
                 flags_fortran=None, \
                 compiler_fortran=None):
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

        """
        # ...
        self._flags            = flags
        self._flags_fortran    = flags_fortran
        self._compiler_fortran = compiler_fortran
        self._prefix           = prefix
        self._path             = path
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
        if compiler_fortran:
            if "mpi" in compiler_fortran.lower():
                args += ['-DMPI_ENABLED=ON']
        # ...

        # ...
        if prefix:
            if not os.path.exists(prefix):
                mkdir_p(prefix)
            prefix = os.path.abspath(prefix)

            args += [' -DCMAKE_INSTALL_PREFIX={0}'.format(prefix)]

        if flags_fortran:
            args += [' -DCMAKE_Fortran_FLAGS="{0}"'.format(flags_fortran)]
#            args += [' -DCMAKE_Fortran_FLAGS_DEBUG={0}'.format(flags_fortran)]
#            args += [' -DCMAKE_Fortran_FLAGS_RELEASE={0}'.format(flags_fortran)]
        # ...

        # ...
        if flags:
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

    def initialize(self, src_dir, project, suffix, libname, force=True):

        # ...
        from pyccel import codegen
        codegen_dir  = os.path.dirname(os.path.realpath(str(codegen.__file__)))
        templates_dir = os.path.join(codegen_dir, 'templates')

        cmakemodules_src = os.path.join(templates_dir, 'cmake')
        cmakelists_src   = os.path.join(templates_dir, 'CMakeLists.txt')
        package_src      = os.path.join(templates_dir, 'package')

        cmakemodules_dst = os.path.join(src_dir, 'cmake')
        cmakelists_dst   = os.path.join(src_dir, 'CMakeLists.txt')
        package_dst      = os.path.join(src_dir, 'package')
        # ...

        # ...
        import shutil, errno
        def _copydata(src, dst):
            try:
                # TODO improve
                if os.path.exists(dst):
                    os.system('rm -rf {0}'.format(dst))

                shutil.copytree(src, dst, ignore=shutil.ignore_patterns('*.pyc'))

            except OSError as exc: # python >2.5
                if exc.errno == errno.ENOTDIR:
                    shutil.copy(src, dst)
                else: raise

                if not os.path.exists(src_dir):
                    raise ValueError('Could not find :{0}'.format(src_dir))
        # ...

        # ... update CMakeLists.txt
        def _print_cmakelists(src, dst):
            f = open(src, 'r')
            code = f.readlines()
            f.close()

            code = ''.join(l for l in code)

            code = code.replace('__PROJECT__', project)
            code = code.replace('__SRC_DIR__', src_dir)
            code = code.replace('__SUFFIX__',  suffix)
            code = code.replace('__LIBNAME__', libname)

            if force or (not os.path.isfile(dst)):
                f = open(dst, 'w')
                f.write(code)
                f.close()
        # ...

        # ... update make_package.py
        def _print_make_package(package_src, package_dst):

            src = os.path.join(package_src, 'make_package.py')
            dst = os.path.join(package_dst, 'make_package.py')

            f = open(src, 'r')
            code = f.readlines()
            f.close()

            code = ''.join(l for l in code)

            code = code.replace('__PROJECT__', project)
            code = code.replace('__SUFFIX__',  suffix.upper())
            code = code.replace('__LIBNAME__', libname)

            if force or (not os.path.isfile(dst)):
                f = open(dst, 'w')
                f.write(code)
                f.close()
        # ...

        # ... TODO: uncomment add_subdirectory(package) from templates/CMakeLists.txt
        _print_cmakelists(cmakelists_src, cmakelists_dst)
        # ...

        # ...
        _copydata(cmakemodules_src, cmakemodules_dst)
        _copydata(package_src, package_dst)
        # ...

        # ...
        src = os.path.join(package_src, 'CMakeLists.txt')
        dst = os.path.join(package_dst, 'CMakeLists.txt')

        _print_cmakelists(src, dst)
        # ...

        # ...
        _print_make_package(package_src, package_dst)
        # ...
