# coding: utf-8

import os
import importlib

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

    # ... clapp environment
    pyccel_modules += ['plaf', 'spl', 'disco', 'fema']
    # ...

    # ... TODO add only if used
#    user_modules = ['m_pyccel']
    user_modules = []
    # ...

    # ...
    d = find_imports(filename=filename)

    imports = {}

    ignored_modules.append('pyccel')
    for n in pyccel_modules:
        ignored_modules.append('pyccel.{0}'.format(n))
        ignored_modules.append(n)

    # TODO remove. for the moment we use 'from spl.bspline import *'
    ignored_modules.append('plf')
    ignored_modules.append('dsc')
    ignored_modules.append('jrk')

    # ...
    def _ignore_module(key):
        for i in ignored_modules:
            if i == key:
                return True
            else:
                n = len(i)
                if i == key[:n]:
                    return True
        return False
    # ...

    for key, value in list(d.items()):
        if not _ignore_module(key):
            imports[key] = value
    ms = []
    for module, names in list(imports.items()):
        codegen_m = FCodegen(filename=module+".py", name=module, is_module=True,
                            output_dir=output_dir)
        codegen_m.doprint(language=language, accelerator=accelerator, \
                          ignored_modules=ignored_modules, \
                          with_mpi=with_mpi)
        ms.append(codegen_m)

    codegen = FCodegen(filename=filename, name=name, output_dir=output_dir)
    s=codegen.doprint(language=language, accelerator=accelerator, \
                      ignored_modules=ignored_modules, with_mpi=with_mpi, \
                      pyccel_modules=pyccel_modules, \
                      user_modules=user_modules)

    if show:
        print('========Fortran_Code========')
        print(s)
        print('============================')
        print((">>> Codegen :", name, " done."))

    modules = codegen.modules
    # ...

    # ...
    if single_file:
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

        ls = ms + [codegen]
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
    if compiler:
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
def build_cmakelists(src_dir, libname, files, force=True, dep_libs=[]):
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
    # ...

    # ...
    code = ''
    code = '{0}\n{1}'.format(code, _print_files(files))
    code = '{0}\n{1}'.format(code, _print_libname(libname))
    code = '{0}\n{1}'.format(code, _print_dependencies(dep_libs))
    # ...

    setup_path = os.path.join(src_dir, 'CMakeLists.txt')
    if force or (not os.path.isfile(setup_path)):
        # ...
        f = open(setup_path, 'w')
        f.write(code)
        f.close()
        # ...
# ...

# ...
def load_extension(ext, output_dir, clean=True, modules=None, silent=True):
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
    """

    # ...
    base_dir = output_dir
    output_dir = os.path.join(base_dir, ext)
    mkdir_p(output_dir)
    # ...

    # ...
    extension = 'pyccelext_{0}'.format(ext)
    try:
        package = importlib.import_module(extension)
    except:
        raise ImportError('could not import {0}'.format(extension))
    ext_dir = str(package.__path__[0])
    # ...

    # ...
    if not modules:
        py_file     = lambda f: (f.split('.')[-1] == 'py')
        ignore_file = lambda f: (os.path.basename(f) in ['__init__.py'])

        files = [f for f in os.listdir(ext_dir) if py_file(f) and not ignore_file(f)]
        modules = [f.split('.')[0] for f in files]
    elif isinstance(modules, str):
        modules = [modules]
    # ...

    for module in modules:
        try:
            m = importlib.import_module(extension, package=module)
        except:
            raise ImportError('could not import {0}.{1}'.format(extension, module))

        m = getattr(m, '{0}'.format(module))

        # remove 'c' from *.pyc
        filename = m.__file__[:-1]

        if not silent:
            print ('> converting {0}/{1}'.format(ext, os.path.basename(filename)))

        build_file(filename, language='fortran', compiler=None, output_dir=output_dir)


    # remove .pyccel temporary files
    if clean:
        os.system('rm {0}/*.pyccel'.format(output_dir))

    # create CMakeLists.txt for the extension
    # TODO add here valid files extensions
    valid_file = lambda f: (f.split('.')[-1] in ['f90'])

    files = [f for f in os.listdir(output_dir) if valid_file(f)]

    libname = extension
    # TODO add dependencies
    dep_libs = []
    build_cmakelists(output_dir, libname=libname,
                     files=files, dep_libs=dep_libs)
# ...
