# coding: utf-8

# TODO: - parse/codegen dependencies only if a flag is True
#       - use unique on files in print cmakelists

import os
import importlib
import numpy as np
from shutil import copyfile


from pyccel.ast.core import DottedName
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
def construct_tree(filename, ignored_modules):
    """Constructs a tree of dependencies given a file to process."""
    # ...
    def _ignore_module(key):
        pattern = lambda s: '{0}.'.format(s)
        return np.asarray([key == pattern(i) for i in ignored_modules]).any()
    # ...

    # ... filters
    is_external_submodule  = lambda m,s: m == 'pyccelext.{0}.external.{1}'.format(ext, s)
    is_extension_submodule = lambda m,s: m == 'pyccelext.{0}.{1}'.format(ext, s)

    is_parallel_submodule  = lambda m,s: m == 'pyccel.stdlib.parallel.{0}'.format(s)
    is_stdlib_external_submodule  = lambda m,s: m == 'pyccel.stdlib.external.{0}'.format(s)
    # ...

    # ... parse imports within the current file
    d = find_imports(filename=filename)

    imports = {}
    for key, value in list(d.items()):
        if not _ignore_module(key):
            imports[key] = value
    # ...

    # ...
    imports_src = {}
    for module, names in list(imports.items()):
#        print('> module, names = ', module, names)
        f_names = []
        for n in names:
#            print('> n = ', n)
            if module.startswith('pyccelext'):
                ext_full  = module.split('pyccelext.')[-1]
                ext       = ext_full.split('.')[0] # to remove submodule
                if module == 'pyccelext.{0}'.format(ext):
                    ext_dir = get_extension_path(ext)
                    # TODO import all files within a package

                    f_name = 'pyccelext_{0}.py'.format(n)
                else:
                    submodule = ext_full.split('.')[-1] # to get submodule

                    # TODO 'elif' test is wrong
                    if is_extension_submodule(module, submodule):
                        f_name = get_extension_path(ext, module=submodule)
                    elif is_external_submodule(module, submodule):
                        f_name = get_extension_path(ext, module=submodule, is_external=True)
                    else:
                        raise ValueError('non valid import for pyccel extensions.')
            elif module.startswith('pyccel.stdlib.parallel'):
                ext_full  = module.split('pyccel.stdlib.parallel.')[-1]
                ext       = ext_full.split('.')[0] # to remove submodule

                submodule = ext_full.split('.')[-1] # to get submodule
#                print(ext, submodule)

                if is_parallel_submodule(module, submodule):
                    f_name = get_parallel_path(ext, module=submodule)
                else:
                    raise ValueError('non valid import for parallel pyccel package.')
            elif module.startswith('pyccel.stdlib.external'):
                ext_full  = module.split('pyccel.stdlib.external.')[-1]
                ext       = ext_full.split('.')[0] # to remove submodule

                submodule = ext_full.split('.')[-1] # to get submodule

                if is_stdlib_external_submodule(module, submodule):
                    f_name = get_stdlib_external_path(ext, module=submodule)
                else:
                    raise ValueError('non valid import for pyccel stdlib external package.')
            else:
                filename_py  = '{0}.py'.format(module)
                filename_pyh = '{0}.pyh'.format(module)

                if os.path.isfile(filename_py):
                    f_name = filename_py
                elif os.path.isfile(filename_pyh):
                    f_name = filename_pyh
                else:
                    raise ValueError('Could not find '
                                     '{0} or {1}'.format(filename_py, filename_pyh))

            if isinstance(f_name, str):
                f_names.append(f_name)
            elif isinstance(f_name, (list, tuple)):
                f_names += list(f_name)
            else:
                raise TypeError('Expecting a str, tuple or list')

        # this is to avoid duplication in filenames,
        # we avoid using sets here, to respect the imports order
        imports_src[module] = []
        for f_name in f_names:
            if not(f_name in imports_src[module]):
                # we don't process header files
                if f_name.endswith('.py'):
                    ims, ims_src = construct_tree(f_name, ignored_modules)
                    for m, ns in ims.items():
                        if m in imports:
                            imports[m] += ns
                        else:
                            imports[m]  = ns

                        if m in imports_src:
                            imports_src[m] += ims_src[m]
                        else:
                            imports_src[m]  = ims_src[m]

                imports_src[module] += [f_name]
    #...

    # TODO must use ordered dictionaries from here

    return imports, imports_src
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
        One among ('openmp', 'openacc')
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
    if not name:
        name = os.path.basename(filename.split('.')[0])
    # ...

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

    # ...
    user_modules = []
    # ...

    # ...
    # ignoring pyccel.stdlib import
    ignored_modules.append('pyccel.stdlib')
    # ...

    # ...
    imports, imports_src = construct_tree(filename, ignored_modules)
    # ...

    # ...
    namespaces = {}
    namespace_user = {}
    namespace_user['cls_constructs'] = {}

    # we store here the external library dependencies for every module
    d_libraries = []

    ms = []
    treated_files = []
    for module, names in list(imports.items()):
        if not(module in namespaces):
            namespaces[module] = {}

#        print('>>> processing module:{0}, and names:{1}'.format(module, names))

        f_names = imports_src[module]
        for f_name in f_names:
            if f_name in treated_files:
                break

#            print('> treating {0}'.format(f_name))

            module_name = str(module).replace('.', '_')

            codegen_m = FCodegen(filename=f_name,
                                 name=module_name,
                                 output_dir=output_dir)
            codegen_m.doprint(language=language,
                              accelerator=accelerator,
                              ignored_modules=ignored_modules,
                              with_mpi=with_mpi)
            _append_module = True
            if '__ignore_at_import__' in codegen_m.metavars:
                if codegen_m.metavars['__ignore_at_import__']:
                    ignored_modules.append(module)
                    _append_module = False

            if '__libraries__' in codegen_m.metavars:
                deps = codegen_m.metavars['__libraries__'].split(',')
                d_libraries += deps

            if _append_module:
                ms.append(codegen_m)

            for k,v in codegen_m.namespace.items():
                namespaces[module][k] = v

            cls_constructs = namespaces[module].pop('cls_constructs', {})

            avail_names = set(namespaces[module].keys())
            avail_names = set(names).intersection(avail_names)
            for n in avail_names:
                namespace_user[n] = namespaces[module][n]

            for k,v in cls_constructs.items():
                namespace_user['cls_constructs'][k] = v

            from pyccel.parser.syntax.core import print_namespace
            from pyccel.parser.syntax.core import update_namespace
            from pyccel.parser.syntax.core import get_namespace

            update_namespace(namespace_user)
            treated_files += [f_name]
#            print_namespace()

            # TODO add aliases or what to import (names)
#    import sys; sys.exit(0)


    codegen = FCodegen(filename=filename,
                       name=name,
                       output_dir=output_dir)
    s=codegen.doprint(language=language,
                      accelerator=accelerator,
                      ignored_modules=ignored_modules,
                      with_mpi=with_mpi,
                      pyccel_modules=pyccel_modules,
                      user_modules=user_modules)


    if '__libraries__' in codegen.metavars:
        deps = codegen.metavars['__libraries__'].split(',')
        d_libraries += deps

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
        if single_file:
            ls = ms + [codegen]
        else:
            ls = [codegen]

        codes = [m.code for m in ls]
        # ...

        # ...
        f = open(codegen.filename_out, "w")

        for line in pyccel_code:
            f.write(line)

        for code in codes:
            for line in code:
                f.write(line)
            f.write('\n')

        f.close()
        # ...
    # ...

    # ...
    if compiler and (not codegen.is_header):
        for codegen_m in ms:

            # we only compile a module if it is non empty
            is_valid = len(codegen_m.body.strip() +
                           codegen_m.routines.strip()  +
                           codegen_m.classes.strip() ) > 0

            if is_valid:
                if str(codegen_m.name).startswith('m_pyccel_stdlib_'):
                    break

#                print('>>>>>> {0}'.format(codegen_m.filename))
                compiler_m = Compiler(codegen_m,
                                      compiler=compiler,
                                      accelerator=accelerator,
                                      debug=debug,
                                      include=include,
                                      libdir=libdir,
                                      libs=libs)
                compiler_m.compile(verbose=verbose)

        # TODO ARA : to remove
        ignored_modules += ['pyccel.stdlib.parallel.mpi_new']
        c = Compiler(codegen,
                     compiler=compiler,
                     inline=inline,
                     accelerator=accelerator,
                     debug=debug,
                     include=include,
                     libdir=libdir,
                     libs=libs,
                     ignored_modules=ignored_modules)
        c.compile(verbose=verbose)

        if execute:
            execute_file(c.binary)
    # ...

    # ...
    info = {}
    info['namespaces'] = namespaces
    info['libs']   = d_libraries
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
                     libs=[],
                     programs=[]):
    # ...
    def _print_files(files):
        files_str = ' '.join(i for i in files)
        return 'set(files {0})'.format(files_str)

    def _print_libname(libname):
        return 'add_library({0} {1})'.format(libname, '${files}')

    def _print_dependencies(libs):
        if len(libs) == 0:
            return ''
        deps_str  = ' '.join(i for i in libs)
        return 'TARGET_LINK_LIBRARIES({0} {1})'.format(libname, deps_str)

    def _print_programs(programs, libname, libs):
        if len(programs) == 0:
            return ''

        code_deps = ' '.join(i for i in libs)
        code = ''
        for f in programs:
            name = f.split('.')[0] # file name without extension

            code_bin = '{0}_{1}'.format(libname, name)

            code += '\n# ... {0}\n'.format(f)
            code += 'ADD_EXECUTABLE({0} {1})\n'.format(code_bin, f)

            if len(libs) > 0:
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
        code = '{0}\n{1}'.format(code, _print_dependencies(libs))

        libs += [libname]

    code = '{0}\n{1}'.format(code, _print_programs(programs, libname, libs))
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
                             libs=[]):
    # ...
    if (len(files) == 0):
        return
    # ...

    # ...
    code_deps = ' '.join(i for i in libs)
    code = ''
    for f in files:
        name = f.split('.')[0] # file name without extension

        code_bin = 'test_{0}_{1}'.format(project, name)

        code += '\n# ... {0}\n'.format(f)
        code += 'ADD_EXECUTABLE({0} {1})\n'.format(code_bin, f)

        if len(libs) > 0:
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
def get_parallel_path(ext, module=None):
    """Finds the path of a pyccel parallel package (.py or .pyh).
    A specific module can also be given."""

    extension = 'pyccel.stdlib.parallel'
    try:
        package = importlib.import_module(extension)
    except:
        raise ImportError('could not import {0}'.format(extension))

    ext_dir = str(package.__path__[0])

    if not module:
        return ext_dir

    filename_py  = '{0}.py'.format(module)
    filename_pyh = '{0}.pyh'.format(module)

    filename_py  = os.path.join(ext_dir, filename_py)
    filename_pyh = os.path.join(os.path.join(ext_dir, 'external'), filename_pyh)

    if not os.path.isfile(filename_py) and not os.path.isfile(filename_pyh):
        raise ImportError('could not find {0} or {1}'.format(filename_py,
                                                             filename_pyh))

    files = []
    if os.path.isfile(filename_py):
        files.append(filename_py)
    if os.path.isfile(filename_pyh):
        files.append(filename_pyh)

    return files
# ...

# ...
def get_stdlib_external_path(ext, module=None):
    """Finds the path of a pyccel stdlib external package (.py or .pyh).
    A specific module can also be given."""

    extension = 'pyccel.stdlib.external'
    try:
        package = importlib.import_module(extension)
    except:
        raise ImportError('could not import {0}'.format(extension))

    ext_dir = str(package.__path__[0])

    if not module:
        return ext_dir

    filename_py  = '{0}.py'.format(module)
    filename_pyh = '{0}.pyh'.format(module)

    filename_py  = os.path.join(ext_dir, filename_py)
    filename_pyh = os.path.join(ext_dir, filename_pyh)

    if os.path.isfile(filename_py):
        return filename_py
    elif os.path.isfile(filename_pyh):
        return filename_pyh
    else:
        raise ImportError('could not find {0} or {1}'.format(filename_py,
                                                             filename_pyh))
# ...

# ...
def get_extension_path(ext, module=None, is_external=False):
    """Finds the path of a pyccel extension (.py or .pyh).
    A specific module can also be given."""

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

    filename_py  = '{0}.py'.format(module)
    filename_pyh = '{0}.pyh'.format(module)

    if not is_external:
        filename_py  = os.path.join(ext_dir, filename_py)
        filename_pyh = os.path.join(ext_dir, filename_pyh)
    else:
        filename_py  = os.path.join(ext_dir, filename_py)
        filename_pyh = os.path.join(os.path.join(ext_dir, 'external'), filename_pyh)

    if os.path.isfile(filename_py):
        return filename_py
    elif os.path.isfile(filename_pyh):
        return filename_pyh
    else:
        raise ImportError('could not find {0} or {1}'.format(filename_py,
                                                             filename_pyh))
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

    return '{0}/testing'.format(ext_dir)
# ...

# ...
def get_extension_external_path(ext):
    """Finds the path of a pyccel extension external files.

    an external file to pyccel, is any low level code associated to its
    header(s)."""

    extension = 'pyccelext_{0}'.format(ext)
    try:
        package = importlib.import_module(extension)
    except:
        raise ImportError('could not import {0}'.format(extension))

    ext_dir = str(package.__path__[0])

    return '{0}/external'.format(ext_dir)
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

    # ... TODO create a list for valid external files
    ignore_file   = lambda f: (os.path.basename(f) in ['__init__.py'])
    py_file       = lambda f: (f.split('.')[-1] == 'py')
    pyh_file      = lambda f: (f.split('.')[-1] == 'pyh')
    external_file = lambda f: (f.split('.')[-1] in ['f90', 'F90', 'f'])
    # ...

    # ...
    if not modules:
        files = [f for f in os.listdir(ext_dir) if py_file(f) and not ignore_file(f)]
        modules = [f.split('.')[0] for f in files]
    elif isinstance(modules, str):
        modules = [modules]
    # ...

    # ... convert all modules of the seleceted extension
    infos = {}
    for module in modules:
        filename = get_extension_path(ext, module=module)
        if not silent:
            f_name = os.path.basename(filename)
            print ('> converting extensions/{0}/{1}'.format(ext, f_name))

        module_name = 'pyccelext_{0}_{1}'.format(ext, f_name.split('.py')[0])

        infos[module] = build_file(filename,
                                   language=language,
                                   compiler=None,
                                   output_dir=output_dir,
                                   name=module_name,
                                   single_file=False)
    # ...

    # remove .pyccel temporary files
    if clean:
        os.system('rm {0}/*.pyccel'.format(output_dir))

    # ... create CMakeLists.txt for the extension
    #     TODO add here valid files extensions
    valid_file = lambda f: (f.split('.')[-1] in ['f90'])
    files = [f for f in os.listdir(output_dir) if valid_file(f)]
    # ...

    # ... update files list for external files
    if os.path.isdir(os.path.join(ext_dir, 'external')):
        external_dir = get_extension_external_path(ext)

        external_files = [f for f in os.listdir(external_dir) if
                          external_file(f) and not ignore_file(f)]

        files = external_files + files

        # copy external files to output_dir
        for f_name in external_files:
            f_src = os.path.join(external_dir, f_name)
            f_dst = os.path.join(output_dir, f_name)

            copyfile(f_src, f_dst)
    # ...

    # ... construct external library dependencies
    libs = []
    for module, info in infos.items():
        deps = [i.strip() for i in info['libs']]
        libs += deps
    # TODO must remove duplicated libs?
    # ...

    # ...
    libname = extension
    build_cmakelists(output_dir,
                     libname=libname,
                     files=files,
                     libs=libs)
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

            infos[filename] = build_file(filename,
                                         language=language,
                                         compiler=None,
                                         single_file=False,
                                         output_dir=output_testing_dir)

        # ... construct external library dependencies
        libs = []
        for module, info in infos.items():
            deps = [i.strip() for i in info['libs']]
            libs += deps
        # TODO must remove duplicated libs?
        # ...

        # remove .pyccel temporary files
        if clean:
            os.system('rm {0}/*.pyccel'.format(output_testing_dir))

        valid_file = lambda f: (f.split('.')[-1] in ['f90'])
        files = [f for f in os.listdir(output_testing_dir) if valid_file(f)]

        libs += [libname]
        build_testing_cmakelists(output_testing_dir, ext,
                                 files=files,
                                 libs=libs)
    # ...

    # ...
    build_cmakelists_dir(base_dir, testing=testing)
    # ...
# ...

# ... # default values of FC etc must be done in settings before
def initialize_project(base_dir, project, libname, settings):

    # ...
    if not os.path.exists(base_dir):
        raise ValueError('Could not find :{0}'.format(base_dir))
    # ...

    # ...
    prefix = settings['prefix']
    fflags = settings['fflags']
    flags  = settings['flags']
    fc     = settings['fc']
    suffix = settings['suffix']
    # ...

    # ...
    mkdir_p(prefix)
    # ...

    # ...
    from pyccel.codegen.cmake import CMake
    cmake = CMake(base_dir, \
                  prefix=prefix, \
                  flags=flags, \
                  flags_fortran=fflags, \
                  compiler_fortran=fc)

    cmake.initialize(base_dir, project, suffix, libname, force=True)

    cmake.configure()
    cmake.make()

    # TODO uncomment install
    #cmake.install()
    # ...
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
        code_header = '# ... pyccel extensions are listed below'
        code = '{0}\n{1}'.format(code, code_header)

        extensions = ', '.join(i for i in settings['extensions'])
        code_ext = 'extensions = ["{0}"]'.format(extensions)
        code = '{0}\n{1}\n'.format(code, code_ext)

        code = ('{0}'
                '# ...\n').format(code)
    # ...

    # ... add flags
    if settings['flags']:
        code_header = '# ... pyccel flags are listed below'
        code = '{0}\n{1}'.format(code, code_header)

        flags = ', '.join('\n"{0}": "{1}"'.format(k,v)
                          for k,v in settings['flags'].items())
        code_flags = 'flags = {%s\n}' % flags
        code = '{0}\n{1}\n'.format(code, code_flags)

        code = ('{0}'
                '# ...\n').format(code)
    # ...

    # ... add compiler flags
    if settings['fflags']:
        code_header = '# ... compiler flags are listed below'
        code = '{0}\n{1}\n'.format(code, code_header)

        code = ('{0}'
                'fflags = "{1}"\n').format(code, settings['fflags'])

    if settings['fc']:
        code = ('{0}'
                'fc = "{1}"\n').format(code, settings['fc'])

    if settings['suffix']:
        code = ('{0}'
                'suffix = "{1}"\n').format(code, settings['suffix'])

        code = ('{0}'
                '# ...\n').format(code)
    # ...

    # ... add prefix
    if settings['prefix']:
        code_header = '# ... installation prefix '
        code = '{0}\n{1}\n'.format(code, code_header)

        code = ('{0}'
                'prefix = "{1}"\n').format(code, settings['prefix'])

        code = ('{0}'
                '# ...\n').format(code)
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

