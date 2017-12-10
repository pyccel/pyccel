# coding: utf-8

# TODO: - parse/codegen dependencies only if a flag is True
#       - use unique on files in print cmakelists

import os
import importlib
import numpy as np

from pyccel.parser.utilities import find_imports
from pyccel.codegen.codegen  import FCodegen
from pyccel.codegen.compiler import Compiler

# ...
# TODO improve, when using mpi
def execute_file(binary):
    """
    Execute a binary file.

    binary: str
        the name of the binary file
    """
    cmd = binary
    if not ('/' in binary):
        cmd = "./" + binary
    os.system(cmd)
# ...

# ...
def mkdir_p(dir):
    # type: (unicode) -> None
    if os.path.isdir(dir):
        return
    os.makedirs(dir)
# ...

# ...
def build_file(filename, language, compiler, \
               execute=False, accelerator=None, \
               debug=False, verbose=False, show=False, \
               inline=False, name=None, \
               output_dir=None, \
               ignored_modules=['numpy', 'scipy', 'sympy'], \
               pyccel_modules=[], \
               include=[], libdir=[], libs=[], \
               single_file=True):
    """
    User friendly interface for code generation.

    filename: str
        name of the file to load.
    language: str
        low-level target language used in the conversion
    compiler: str
        used compiler for the target language.
    execute: bool
        execute the generated code, after compiling if True.
    accelerator: str
        name of the selected accelerator.
        For the moment, only 'openmp' is available
    debug: bool
        add some useful prints that may help for debugging.
    verbose: bool
        talk more
    show: bool
        prints the generated code if True
    inline: bool
        set to True, if the file is being load inside a python session.
    name: str
        name of the generated module or program.
        If not given, 'main' will be used in the case of a program, and
        'pyccel_m_${filename}' in the case of a module.
    ignored_modules: list
        list of modules to ignore (like 'numpy', 'sympy').
        These modules do not have a correspondence in Fortran.
    pyccel_modules: list
        list of modules supplied by the user.
    include: list
        list of include directories paths
    libdir: list
        list of lib directories paths
    libs: list
        list of libraries to link with

    Example

    >>> from pyccel.codegen import build_file
    >>> code = '''
    ... n = int()
    ... n = 10
    ...
    ... x = int()
    ... x = 0
    ... for i in range(0,n):
    ...     for j in range(0,n):
    ...         x = x + i*j
    ... '''
    >>> filename = "test.py"
    >>> f = open(filename, "w")
    >>> f.write(code)
    >>> f.close()
    >>> build_file(filename, "fortran", "gfortran", show=True, name="main")
    ========Fortran_Code========
    program main
    implicit none
    integer :: i
    integer :: x
    integer :: j
    integer :: n
    n = 10
    x = 0
    do i = 0, n - 1, 1
        do j = 0, n - 1, 1
            x = i*j + x
        end do
    end do
    end
    ============================
    """
#    print('>>>> calling build_file for {0}'.format(filename))
    # ...
    with_mpi = False
    if compiler:
        if 'mpi' in compiler:
            with_mpi = True
    # ...

    # ...
    if with_mpi:
        pyccel_modules.append('mpi')
    # ...

    # ... TODO add only if used
#    user_modules = ['m_pyccel']
    user_modules = []
    # ...

    # ...
    imports = {}

    # ignoring pyccel.stdlib import
    ignored_modules.append('pyccel.stdlib')

    ignored_modules.append('pyccel')
    for n in pyccel_modules:
        ignored_modules.append('pyccel.{0}'.format(n))
        ignored_modules.append(n)

    # ...
    def _ignore_module(key):
        pattern = lambda s: '{0}.'.format(s)

        return np.asarray([key.startswith(pattern(i)) for i in ignored_modules]).any()
    # ...

    d = find_imports(filename=filename)
    for key, value in list(d.items()):
        if not _ignore_module(key):
            imports[key] = value

    imports_src = {}
    for module, names in list(imports.items()):
        f_names = []
        for n in names:
            if module.startswith('pyccelext'):
                ext_full  = module.split('pyccelext.')[-1]
                ext       = ext_full.split('.')[0] # to remove submodule
                if module == 'pyccelext.{0}'.format(ext):
                    ext_dir = get_extension_path(ext)
                    # TODO import all files within a package

                    f_name = 'pyccelext_{0}.py'.format(n)
                else:
                    submodule = ext_full.split('.')[-1] # to get submodule
                    if module == 'pyccelext.{0}.{1}'.format(ext, submodule):
                        f_name = get_extension_path(ext, module=submodule)
                    else:
                        raise ValueError('non valid import for pyccel extensions.')
            else:
                f_name = '{0}.py'.format(n)
            f_names.append(f_name)
        imports_src[module] = f_names

#    print ignored_modules
#    print d
#    print '>>>> imports : ', imports
#    if len(d) > 0:
#        import sys; sys.exit(0)
    #...

    # ...
    namespaces = {}
    namespace_user = {}
    ms = []
    for module, names in list(imports.items()):
        f_name = imports_src[module][0] #TODO loop over files
        codegen_m = FCodegen(filename=f_name,
                             name=module,
                             is_module=True,
                             output_dir=output_dir)
        codegen_m.doprint(language=language,
                          accelerator=accelerator,
                          ignored_modules=ignored_modules,
                          with_mpi=with_mpi)
        ms.append(codegen_m)
        namespaces[module] = codegen_m.namespace
        for n in names:
            namespace_user[n] = namespaces[module][n]

        # TODO add aliases or what to import (names)

    from pyccel.parser.syntax.core import update_namespace
    update_namespace(namespace_user)

    codegen = FCodegen(filename=filename,
                       name=name,
                       output_dir=output_dir)
    s=codegen.doprint(language=language,
                      accelerator=accelerator,
                      ignored_modules=ignored_modules,
                      with_mpi=with_mpi,
                      pyccel_modules=pyccel_modules,
                      user_modules=user_modules)

    # ... TODO shall we use another key?
    namespaces[filename] = codegen.namespace
    # ...

    if show and (not codegen.is_header):
        print('========Fortran_Code========')
        print(s)
        print('============================')

    modules = codegen.modules
    # ...

    # ... TODO improve and remove single_file
    if single_file:
        if not codegen.is_header:
            # ... create a Module for pyccel extra definitions
            pyccel_vars    = []
            pyccel_funcs   = []
            pyccel_classes = []

            stmts = codegen.ast.extra_stmts
            pyccel_funcs   = [i for i in stmts if isinstance(i, FunctionDef)]
            pyccel_classes = [i for i in stmts if isinstance(i, ClassDef)]

            pyccel_stmts  = [i for i in stmts if isinstance(i, Module)]
            if pyccel_vars or pyccel_funcs or pyccel_classes:
                pyccel_stmts += [Module('m_pyccel', \
                                        pyccel_vars, \
                                        pyccel_funcs, \
                                        pyccel_classes)]

            pyccel_code = ''
            for stmt in pyccel_stmts:
                pyccel_code += codegen.printer(stmt) + "\n"
            # ...

            # ...
            f = open(codegen.filename_out, "w")

            for line in pyccel_code:
                f.write(line)

            # this commented line must be removed.
            # otherwise, we will print all used modules in the current file
            #ls = ms + [codegen]
            ls = [codegen]
            codes = [m.code for m in ls]
            for code in codes:
                for line in code:
                    f.write(line)

            f.close()
            # ...
    else:
        raise NotImplementedError('single_file must be True')
    # ...

    # ...
    if compiler and (not codegen.is_header):
        for codegen_m in ms:
            compiler_m = Compiler(codegen_m, \
                                  compiler=compiler, \
                                  accelerator=accelerator, \
                                  debug=debug, \
                                  include=include, \
                                  libdir=libdir, \
                                  libs=libs)
            compiler_m.compile(verbose=verbose)

        c = Compiler(codegen, \
                     compiler=compiler, \
                     inline=inline, \
                     accelerator=accelerator, \
                     debug=debug, \
                     include=include, \
                     libdir=libdir, \
                     libs=libs)
        c.compile(verbose=verbose)

        if execute:
            execute_file(c.binary)
    # ...

    # ...
    info = {}
    info['is_module']  = codegen.is_module
    info['namespaces'] = namespaces
    # ...

    return info
# ...

# ...
# TODO improve args
def load_module(filename, language="fortran", compiler="gfortran"):
    """
    Loads a given filename in a Python session.
    The file will be parsed, compiled and wrapped into python, using f2py.

    filename: str
        name of the file to load.
    language: str
        low-level target language used in the conversion
    compiled: str
        used compiler for the target language.

    Example

    >>> from pyccel.codegen import load_module
    >>> code = '''
    ... def f(n):
    ...     n = int()
    ...     x = int()
    ...     x = 0
    ...     for i in range(0,n):
    ...         for j in range(0,n):
    ...             x = x + i*j
    ...     print("x = ", x)
    ... '''
    >>> filename = "test.py"
    >>> f = open(filename, "w")
    >>> f.write(code)
    >>> f.close()
    >>> module = load_module(filename="test.py")
    >>> module.f(5)
    x =          100
    """
    # ...
    name = filename.split(".")[0]
    name = 'pyccel_m_{0}'.format(name)
    # ...

    # ...
    build_file(filename=filename, language=language, compiler=compiler, \
               execute=False, accelerator=None, \
               debug=False, verbose=True, show=True, inline=True, name=name)
    # ...

    # ...
    try:
        import external
        reload(external)
    except:
        pass
    import external
    # ...

    module = getattr(external, '{0}'.format(name))

    return module
# ...

# ...
def build_cmakelists(src_dir, libname, files,
                     force=True,
                     dep_libs=[],
                     programs=[]):
    # ...
    def _print_files(files):
        files_str = ' '.join(i for i in files)
        return 'set(files {0})'.format(files_str)

    def _print_libname(libname):
        return 'add_library({0} {1})'.format(libname, '${files}')

    def _print_dependencies(dep_libs):
        if len(dep_libs) == 0:
            return ''
        deps_str  = ' '.join(i for i in dep_libs)
        return 'TARGET_LINK_LIBRARIES({0} {1})'.format(libname, deps_str)

    def _print_programs(programs, libname, dep_libs):
        if len(programs) == 0:
            return ''

        code_deps = ' '.join(i for i in dep_libs)
        code = ''
        for f in programs:
            name = f.split('.')[0] # file name without extension

            code_bin = '{0}_{1}'.format(libname, name)

            code += '\n# ... {0}\n'.format(f)
            code += 'ADD_EXECUTABLE({0} {1})\n'.format(code_bin, f)

            if len(dep_libs) > 0:
                code += 'TARGET_LINK_LIBRARIES({0} {1})\n'.format(code_bin, code_deps)

            code += 'ADD_TEST( NAME {0} COMMAND {0} )\n'.format(code_bin)
            code += '# ...\n'

        return code
    # ...

    # ...
    if (len(files) == 0) and (len(programs) == 0):
        return
    # ...

    # ...
    code = ''
    if len(files) > 0:
        code = '{0}\n{1}'.format(code, _print_files(files))
        code = '{0}\n{1}'.format(code, _print_libname(libname))
        code = '{0}\n{1}'.format(code, _print_dependencies(dep_libs))

        dep_libs += [libname]

    code = '{0}\n{1}'.format(code, _print_programs(programs, libname, dep_libs))
    # ...

    setup_path = os.path.join(src_dir, 'CMakeLists.txt')
    if force or (not os.path.isfile(setup_path)):
        # ...
        f = open(setup_path, 'w')
        f.write(code)
        f.close()
        # ...
# ...

# ...
def build_testing_cmakelists(src_dir, project, files,
                             force=True,
                             dep_libs=[]):
    # ...
    if (len(files) == 0):
        return
    # ...

    # ...
    code_deps = ' '.join(i for i in dep_libs)
    code = ''
    for f in files:
        name = f.split('.')[0] # file name without extension

        code_bin = 'test_{0}_{1}'.format(project, name)

        code += '\n# ... {0}\n'.format(f)
        code += 'ADD_EXECUTABLE({0} {1})\n'.format(code_bin, f)

        if len(dep_libs) > 0:
            code += 'TARGET_LINK_LIBRARIES({0} {1})\n'.format(code_bin, code_deps)

        code += 'ADD_TEST( NAME {0} COMMAND {0} )\n'.format(code_bin)
        code += '# ...\n'
    # ...

    setup_path = os.path.join(src_dir, 'CMakeLists.txt')
    if force or (not os.path.isfile(setup_path)):
        # ...
        f = open(setup_path, 'w')
        f.write(code)
        f.close()
        # ...
# ...

# ...
def build_cmakelists_dir(src_dir, force=True, testing=False):
    if not os.path.exists(src_dir):
        raise ValueError('Could not find :{0}'.format(src_dir))

    dirs = [f for f in os.listdir(src_dir)
            if os.path.isdir(os.path.join(src_dir, f))]

    # ...
    def _print_dirs(dirs, testing):
        code = ''
        for d in dirs:
            code + '\n# ...\n'
            code += 'add_subdirectory({0})\n'.format(d)
            if testing:
                code += 'IF(BUILD_TESTING)\n'
                code += '  add_subdirectory({0}/testing)\n'.format(d)
                code += 'ENDIF(BUILD_TESTING)\n'

        return code
    # ...

    # ...
    code = ''
    code = '{0}\n{1}'.format(code, _print_dirs(dirs, testing))
    # ...

    setup_path = os.path.join(src_dir, 'CMakeLists.txt')
    if force or (not os.path.isfile(setup_path)):
        # ...
        f = open(setup_path, 'w')
        f.write(code)
        f.close()
        # ...
# ...

# ...
def get_extension_path(ext, module=None):
    """Finds the path of a pyccel extension. A specific module can also be
    given."""

    extension = 'pyccelext_{0}'.format(ext)
    try:
        package = importlib.import_module(extension)
    except:
        raise ImportError('could not import {0}'.format(extension))

    ext_dir = str(package.__path__[0])

    if not module:
        return ext_dir

    # if module is not None
    try:
        m = importlib.import_module(extension, package=module)
    except:
        raise ImportError('could not import {0}.{1}'.format(extension, module))

    m = getattr(m, '{0}'.format(module))

    # remove 'c' from *.pyc
    filename = m.__file__[:-1]

    return filename
# ...

# ...
def get_extension_testing_path(ext):
    """Finds the path of a pyccel extension tests."""

    extension = 'pyccelext_{0}'.format(ext)
    try:
        package = importlib.import_module(extension)
    except:
        raise ImportError('could not import {0}'.format(extension))

    ext_dir = str(package.__path__[0])

    tests_dir = '{0}/testing'.format(ext_dir)

    return tests_dir
# ...

# ...
def load_extension(ext, output_dir,
                   clean=True,
                   modules=None,
                   silent=True,
                   language='fortran',
                   testing=True):
    """
    Load module(s) from a given pyccel extension.

    ext: str
        a pyccel extension is always of the form pyccelext-xxx where 'xxx' is
        the extension name.
    output_dir: str
        directory where to store the generated files
    clean: Bool
        remove all tempororay files (of extension *.pyccel)
    modules: list, str
        a list of modules or a module. every module must be a string.
    silent: bool
        talk more
    language: str
        target language
    testing: bool
        enable unit tests

    Examples

    >>> load_extension('math', 'extensions', silent=False)
    >>> load_extension('math', 'extensions', modules=['bsplines'])
    >>> load_extension('math', 'extensions', modules='quadratures')
    """

    # ...
    base_dir = output_dir
    output_dir = os.path.join(base_dir, ext)
    mkdir_p(output_dir)
    # ...

    # ...
    extension = 'pyccelext_{0}'.format(ext)
    ext_dir   = get_extension_path(ext)
    # ...

    # ...
    ignore_file = lambda f: (os.path.basename(f) in ['__init__.py'])
    py_file     = lambda f: (f.split('.')[-1] == 'py')
    # ...

    # ...
    if not modules:
        files = [f for f in os.listdir(ext_dir) if py_file(f) and not ignore_file(f)]
        modules = [f.split('.')[0] for f in files]
    elif isinstance(modules, str):
        modules = [modules]
    # ...

    # ... convert all modules of the seleceted extension
    for module in modules:
        filename = get_extension_path(ext, module=module)
        if not silent:
            f_name = os.path.basename(filename)
            print ('> converting extensions/{0}/{1}'.format(ext, f_name))

        module_name = 'm_pyccelext_{0}_{1}'.format(ext, f_name.split('.py')[0])
        build_file(filename,
                   language=language,
                   compiler=None,
                   output_dir=output_dir,
                   name=module_name)
    # ...

    # remove .pyccel temporary files
    if clean:
        os.system('rm {0}/*.pyccel'.format(output_dir))

    # ... create CMakeLists.txt for the extension
    #     TODO add here valid files extensions
    valid_file = lambda f: (f.split('.')[-1] in ['f90'])
    files = [f for f in os.listdir(output_dir) if valid_file(f)]

    libname = extension
    # TODO add dependencies
    dep_libs = []
    build_cmakelists(output_dir,
                     libname=libname,
                     files=files,
                     dep_libs=dep_libs)
    # ...

    # ...
    if testing:
        tests_dir = get_extension_testing_path(ext)

        files = [f for f in os.listdir(tests_dir) if py_file(f) and not ignore_file(f)]

        for f_name in files:
            if not silent:
                print ('> converting extensions/{0}/testing/{1}'.format(ext, f_name))

            filename = os.path.join(tests_dir, f_name)
            output_testing_dir = os.path.join(output_dir, 'testing')
            mkdir_p(output_testing_dir)

            build_file(filename,
                       language=language,
                       compiler=None,
                       output_dir=output_testing_dir)

        # remove .pyccel temporary files
        if clean:
            os.system('rm {0}/*.pyccel'.format(output_testing_dir))

        valid_file = lambda f: (f.split('.')[-1] in ['f90'])
        files = [f for f in os.listdir(output_testing_dir) if valid_file(f)]

        dep_libs += [libname]
        build_testing_cmakelists(output_testing_dir, ext,
                                 files=files,
                                 dep_libs=dep_libs)
    # ...

    # ...
    build_cmakelists_dir(base_dir, testing=testing)
    # ...

# ...

# ...
def initialize_project(base_dir, project, suffix, libname, prefix=None):
    if not os.path.exists(base_dir):
        raise ValueError('Could not find :{0}'.format(base_dir))

    if prefix is None:
        prefix = os.path.join(base_dir, 'usr')
        mkdir_p(prefix)

    FC     = 'gfortran'
    FLAGS  = {}
    FFLAGS = '-O2 -fbounds-check'

    from pyccel.codegen.cmake import CMake
    cmake = CMake(base_dir, \
                  prefix=prefix, \
                  flags=FLAGS, \
                  flags_fortran=FFLAGS, \
                  compiler_fortran=FC)

    cmake.initialize(base_dir, project, suffix, libname, force=True)

    cmake.configure()
    cmake.make()

    # TODO uncomment install
    #cmake.install()

#        FLAGS  = self.configs['flags']
#
#        FC     = self.configs['fortran']['compiler']
#        FFLAGS = self.configs['fortran']['flags']
# ...

# ...
def generate_project_init(srcdir, project, **settings):
    """Generates a __init__.py file for the project."""
    srcdir = os.path.join(srcdir, project)
    mkdir_p(srcdir)

    # ...
    def _print_version(version):
        if version is None:
            return ''
        elif isinstance(version, str) and (len(version) == 0):
            return ''

        return '__version__ = "{0}"'.format(version)

    code = '# -*- coding: UTF-8 -*-'
    code = '{0}\n{1}'.format(code, _print_version(settings['version']))

    filename = os.path.join(srcdir, '__init__.py')
    f = open(filename, 'w')
    f.write(code)
    f.close()
    # ...

# ...

# ...
def generate_project_main(srcdir, project, extensions, force=True):
    # ...
    from pyccel import codegen
    codegen_dir  = os.path.dirname(os.path.realpath(str(codegen.__file__)))
    templates_dir = os.path.join(codegen_dir, 'templates')
    dst_dir       = os.path.join(srcdir, project)

    src = os.path.join(templates_dir, 'main.py')
    dst = os.path.join(dst_dir, 'main.py')
    # ...

    # ... update main.py
    f = open(src, 'r')
    code = f.readlines()
    f.close()

    code = ''.join(l for l in code)

    #TODO uncomment
#    for ext in extensions:
#        code_ext = 'from pyccelext.{0} import *'.format(ext)
#        code = '{0}\n{1}'.format(code, code_ext)


    code  = 'from pyccelext.math.constants import ppi\n'
    code += 'print(ppi)\n'
    code += 'x = 1\n'
    code += 'print(x)\n'

    if force or (not os.path.isfile(dst)):
        f = open(dst, 'w')
        f.write(code)
        f.close()
    # ...
# ...

# ...
def generate_project_conf(srcdir, project, **settings):
    """Generates a conf.py file for the project."""
    mkdir_p(srcdir)

    # ...
    code = '# -*- coding: UTF-8 -*-\n'
#    code = '{0}\n{1}'.format(code, _print_version(settings['version']))

    # ... add extensions
    if settings['extensions']:
        code_header = '# pyccel extensions are listed below'
        code = '{0}\n{1}'.format(code, code_header)

        extensions = ', '.join(i for i in settings['extensions'])
        code_ext = 'extensions = ["{0}"]'.format(extensions)
        code = '{0}\n{1}\n'.format(code, code_ext)
    # ...

    # ... add applications
    code_header = '# add your applications below'
    code = '{0}\n{1}'.format(code, code_header)

    applications = [project]
    applications = ', '.join(i for i in applications)
    code_ext = 'applications = ["{0}"]'.format(applications)
    code = '{0}\n{1}\n'.format(code, code_ext)
    # ...

    # ...
    filename = os.path.join(srcdir, 'conf.py')
    f = open(filename, 'w')
    f.write(code)
    f.close()
    # ...
# ...

