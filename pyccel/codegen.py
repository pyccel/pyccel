# coding: utf-8

import os

from sympy.core import Tuple

from pyccel.printers import fcode
from pyccel.parser  import PyccelParser
from pyccel.types.ast import subs
from pyccel.types.ast import DataType, DataTypeFactory
from pyccel.types.ast import (Range, Tensor, Block, \
                              For, Assign, Declare, Variable, \
                              NativeRange, NativeTensor, \
                              FunctionHeader, ClassHeader, MethodHeader, \
                              datatype, While, NativeFloat, \
                              EqualityStmt, NotequalStmt, \
                              MultiAssign, AugAssign, \
                              FunctionDef, ClassDef, Sync, Del, Print, Import, \
                              Comment, AnnotatedComment, \
                              IndexedVariable, Slice, If, \
                              ThreadID, ThreadsNumber, \
                              Stencil, Ceil, Break, \
                              Zeros, Ones, Array, ZerosLike, Shape, Len, \
                              Dot, Sign, IndexedElement, Module, DottedName)

from pyccel.openmp.syntax import OpenmpStmt

from pyccel.parallel.mpi import MPI_Tensor
from pyccel.parallel.mpi import mpify

_module_stmt = (Comment, FunctionDef, ClassDef, \
                FunctionHeader, ClassHeader, MethodHeader)

# ...
def clean(filename):
    """
    removes the generated files: .pyccel and .f90

    filename: str
        name of the file to parse.
    """
    name = filename.split('.py')[0]
    for ext in ["f90", "pyccel"]:
        f_name = name + "." + ext
        cmd = "rm -f " + f_name
        os.system(cmd)
# ...

# ...
def make_tmp_file(filename):
    """
    returns a temporary file of extension .pyccel that will be decorated with
    indent/dedent so that textX can find the blocks easily.

    filename: str
        name of the file to parse.
    """
    name = filename.split('.py')[0]
    return name + ".pyccel"
# ...

# ...
def preprocess_as_str(lines):
    """
    The input python file will be decorated with
    indent/dedent so that textX can find the blocks easily.
    This function will write the output code in filename_out.

    lines: str or list
        python code as a string
    """
    if type(lines) == str:
        ls = lines.split("\n")
        lines = []
        for l in ls:
            lines.append(l + "\n")

#    # to be sure that we dedent at the end
#    lines += "\n"

    lines_new = ""

    def delta(line):
        l = line.lstrip(' ')
        n = len(line) - len(l)
        return n

    tab   = 4
    depth = 0
    old_line = ""
    annotated = ""
    for i,line in enumerate(lines):
        n = delta(line)
        is_empty   = (len(line.lstrip()) == 0)
        is_comment = False
        if not is_empty:
            is_comment = (line.lstrip()[0] == '#')
        skipped = is_empty or is_comment
        is_annotated = (line.lstrip()[0:2] == '#$')

        if n == depth * tab + tab:
            depth += 1
            lines_new += "indent" + "\n"
            lines_new += line
        elif not skipped:
            d = n // tab
            if (d > 0) or (n==0):
                old = delta(old_line)
                m = (old - n) // tab
                depth -= m
                for j in range(0, m):
                    lines_new += "dedent" + "\n"
                lines_new += annotated
                annotated = ""
            lines_new += line
        elif is_annotated:
            annotated += line
        else:
            lines_new += line
        if not skipped:
            old_line = line
    for i in range(0, depth):
        lines_new += "dedent" + "\n"
    lines_new += annotated

    return lines_new
# ...

# ...
def preprocess(filename, filename_out):
    """
    The input python file will be decorated with
    indent/dedent so that textX can find the blocks easily.
    This function will write the output code in filename_out.

    filename: str
        name of the file to parse.

    filename_out: str
        name of the temporary file that will be parsed by textX.
    """
    f = open(filename)
    lines = f.readlines()
    f.close()

    lines_new = preprocess_as_str(lines)

    f = open(filename_out, "w")
    for line in lines_new:
        f.write(line)
    f.close()
# ...

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

class Codegen(object):
    """Abstract class for code generation."""
    def __init__(self, \
                 filename=None, \
                 name=None, \
                 imports=None, \
                 preludes=None, \
                 body=None, \
                 routines=None, \
                 classes=None, \
                 modules=None, \
                 is_module=False):
        """Constructor for the Codegen class.

        filename: str
            name of the file to parse.
        name: str
            name of the generated module or program.
            If not given, 'main' will be used in the case of a program, and
            'pyccel_m_${filename}' in the case of a module.
        imports: list
            list of imports statements as strings.
        preludes: list
            list of preludes statements.
        body: list
            list of body statements.
        routines: list
            list of all routines (functions/subroutines) definitions.
        classes: list
            list of all classes definitions.
        modules: list
            list of used modules.
        is_module: bool
            True if the input file will be treated as a module and not a program
        """
        # ... TODO improve once TextX will handle indentation
        clean(filename)

        filename_tmp = make_tmp_file(filename)
        preprocess(filename, filename_tmp)
        filename = filename_tmp
        # ...

        if body is None:
            body = []
        if routines is None:
            routines = []
        if classes is None:
            classes = []
        if modules is None:
            modules = []

        self._filename     = filename
        self._filename_out = None
        self._language     = None
        self._imports      = imports
        self._name         = name
        self._is_module    = is_module
        self._body         = body
        self._preludes     = preludes
        self._routines     = routines
        self._classes      = classes
        self._modules      = modules
        self._printer      = None

    @property
    def filename(self):
        """Returns the name of the file to convert."""
        return self._filename

    @property
    def filename_out(self):
        """Returns the name of the output file."""
        return self._filename_out

    @property
    def code(self):
        """Returns the generated code"""
        return self._code

    @property
    def ast(self):
        """Returns the generated ast"""
        return self._ast

    @property
    def language(self):
        """Returns the used language"""
        return self._language

    @property
    def name(self):
        """Returns the name of the Codegen class"""
        return self._name

    @property
    def imports(self):
        """Returns the imports of the Codegen class"""
        return self._imports

    @property
    def preludes(self):
        """Returns the preludes of the Codegen class"""
        return self._preludes

    @property
    def body(self):
        """Returns the body of the Codegen class"""
        return self._body

    @property
    def routines(self):
        """Returns the routines of the Codegen class"""
        return self._routines

    @property
    def classes(self):
        """Returns the classes of the Codegen class"""
        return self._classes

    @property
    def modules(self):
        """Returns the modules of the Codegen class"""
        return self._modules

    @property
    def is_module(self):
        """Returns True if we are treating a module."""
        return self._is_module

    @property
    def printer(self):
        """Returns the used printer"""
        return self._printer

    def as_module(self):
        """Generate code as a module. Every extension must implement this method."""
        pass

    def as_program(self):
        """Generate code as a program. Every extension must implement this method."""
        pass

    def doprint(self, language, accelerator=None, \
                ignored_modules=[], \
                with_mpi=False, \
                pyccel_modules=[]):
        """Generate code for a given language.

        language: str
            target language. Possible values {"fortran"}
        accelerator: str
            name of the selected accelerator.
            For the moment, only 'openmp' is available
        ignored_modules: list
            list of modules to ignore (like 'numpy', 'sympy').
            These modules do not have a correspondence in Fortran.
        with_mpi: bool
            True if using MPI
        pyccel_modules: list
            list of modules supplied by the user.
        """
        # ...
        filename = self.filename
        imports  = ""
        preludes = ""
        body     = ""
        routines = ""
        classes  = ""
        modules  = []

        namespace    = {}
        declarations = {}
        # ...

        # ...
        if language.lower() == "fortran":
            printer = fcode
        else:
            raise ValueError("Only fortran is available")
        # ...

        # ...
        self._printer = printer
        # ...

        # ... TODO improve. mv somewhere else
        if not (accelerator is None):
            if accelerator == "openmp":
                imports += "use omp_lib\n"
            else:
                raise ValueError("Only openmp is available")
        # ...

        # ...
        if with_mpi:
            imports += "use MPI\n"
        # ...

        # ...
        pyccel = PyccelParser()
        ast    = pyccel.parse_from_file(filename)
        # ...

        # ... TODO improve
        if len(ast.extra_stmts) > 0:
            imports += "use m_pyccel\n"
        # ...

        # ...
        stmts = ast.expr
        # ...

        # ...
        if with_mpi:
            stmts = [mpify(s) for s in stmts]
        # ...

        for stmt in stmts:
            if isinstance(stmt, (Comment, AnnotatedComment)):
                body += printer(stmt) + "\n"
            elif isinstance(stmt, Import):
                name = str(stmt.fil)
                if not(name in ignored_modules):
                    imports += printer(stmt) + "\n"
                    modules += [name]
            elif isinstance(stmt, Declare):
                decs = stmt
            elif isinstance(stmt, (FunctionHeader, ClassHeader, MethodHeader)):
                continue
            elif isinstance(stmt, Assign):
                if not isinstance(stmt.rhs, (Range, Tensor, MPI_Tensor)):
                    body += printer(stmt) + "\n"
                elif isinstance(stmt.rhs, MPI_Tensor):
                    for dec in stmt.rhs.declarations:
                        preludes += printer(dec) + "\n"
                    for s in stmt.rhs.body:
                        body += printer(s) + "\n"
            elif isinstance(stmt, AugAssign):
                body += printer(stmt) + "\n"
            elif isinstance(stmt, MultiAssign):
                body += printer(stmt) + "\n"
            elif isinstance(stmt, (Zeros, Ones, ZerosLike, Array)):
                body += printer(stmt) + "\n"
            elif isinstance(stmt, (Shape, Len)):
                body += printer(stmt) + "\n"
            elif isinstance(stmt, For):
                body += printer(stmt) + "\n"
            elif isinstance(stmt, While):
                body += printer(stmt) + "\n"
            elif isinstance(stmt, If):
                body += printer(stmt) + "\n"
            elif isinstance(stmt, FunctionDef):
                expr = stmt
                if expr.hide:
                    continue
                if len(expr.results) == 1:
                    result = expr.results[0]
                    if result.allocatable or (result.rank > 0):
                        expr = subs(expr, result, str(expr.name))
                sep = separator()
                routines += sep + printer(expr) + "\n" \
                          + sep + '\n'
            elif isinstance(stmt, ClassDef):
                classes += printer(stmt) + "\n"
            elif isinstance(stmt, Print):
                body += printer(stmt) + "\n"
            elif isinstance(stmt, Del):
                # TODO is it ok to put it in the body?
                body += printer(stmt) + "\n"
            elif isinstance(stmt, Sync):
                body += printer(stmt) + "\n"
            elif isinstance(stmt, Stencil):
                body += printer(stmt) + "\n"
            elif isinstance(stmt, list):
                for s in stmt:
                    body += printer(s) + "\n"
            elif isinstance(stmt, Block):
                for s in stmt.body:
                    body += "\n" + printer(s) + "\n"
                for dec in stmt.declarations:
                    preludes += "\n" + printer(dec) + "\n"
            elif isinstance(stmt, Module):
                body += "\n" + printer(stmt) + "\n"
            else:
                if True:
                    print "> uncovered statement of type : ", type(stmt)
                else:
                    raise Exception('Statement not yet handled.')
        # ...

        # ...
        for key, dec in ast.declarations.items():
            if not isinstance(dec.dtype, (NativeRange, NativeTensor)):
                preludes += printer(dec) + "\n"

        def _construct_prelude(stmt):
            preludes = ''
            if isinstance(stmt, (list, tuple, Tuple)):
                for dec in stmt:
                    preludes += _construct_prelude(dec)
            if isinstance(stmt, (For, While)):
                preludes += _construct_prelude(stmt.body)
            if isinstance(stmt, Block):
                for dec in stmt.declarations:
                    preludes += printer(dec) + "\n"
            return preludes

        preludes += _construct_prelude(stmts)
        # ...

        # ...
        if not self.is_module:
            is_module = True
            for stmt in stmts:
                if not(isinstance(stmt, _module_stmt)):
                    is_module = False
                    break
        else:
            is_module = True
        # ...

#        # ...
#        if with_mpi:
#            for stmt in stmts:
#                if isinstance(stmt, _module_stmt)):
#                    is_module = False
#                    break
#        # ...

        # ...
        self._ast       = ast
        self._imports   = imports
        self._preludes  = preludes
        self._body      = body
        self._routines  = routines
        self._classes   = classes
        self._modules   = modules
        self._is_module = is_module
        self._language  = language
        # ...

        # ...
        if is_module:
            code = self.as_module()
        else:
            code = self.as_program()
        # ...

        # ...
        self._code = code
        # ...

        # ...
        ext = get_extension(language)
        self._filename_out = filename.split(".py")[0] + "." + ext
        # ...

        return code


class FCodegen(Codegen):
    """Code generation for Fortran."""
    def __init__(self, *args, **kwargs):
        """
        Constructor for code generation using Fortran.
        """
        super(FCodegen, self).__init__(*args, **kwargs)

    def as_module(self):
        """Generate code as a module."""
        name     = self.name
        imports  = self.imports
        preludes = self.preludes
        body     = self.body
        routines = self.routines
        classes  = self.classes
        modules  = self.modules

        if name is None:
            name = self.filename.split(".")[0]
            name = name.replace('/', '_')
            name = 'm_{0}'.format(name)

        code  = "module " + str(name)     + "\n"
        code += imports                   + "\n"
        code += "implicit none"           + "\n"
        code += preludes                  + "\n"

        if len(routines) > 0:
            code += "contains"            + "\n"
            code += routines              + "\n"
        if len(classes) > 0:
            code += classes               + "\n"
        code += "end module " + str(name) + "\n"

        return code

    def as_program(self):
        """Generate code as a program."""
        name     = self.name
        imports  = self.imports
        preludes = self.preludes
        body     = self.body
        routines = self.routines
        classes  = self.classes
        modules  = self.modules

        if name is None:
            name = 'main'

        code  = "program " + str(name)    + "\n"
        code += imports                   + "\n"
        code += "implicit none"           + "\n"
        code += preludes                  + "\n"

        if len(body) > 0:
            code += body                  + "\n"
        if len(routines) > 0:
            code += "contains"            + "\n"
            code += routines              + "\n"
        if len(classes) > 0:
            code += classes               + "\n"
        code += "end"                     + "\n"

        return code

# TODO improve later
class PyccelCodegen(Codegen):
    """Code generation for the Pyccel Grammar"""
    pass

# ...
def get_extension(language):
    """
    returns the extension of a given language.

    language: str
        low-level target language used in the conversion
    """
    if language == "fortran":
        return "f90"
    else:
        raise ValueError("Only fortran is available")
# ...

# ...
def separator(n=40):
    """
    Creates a separator string.
    This is used to improve the readability of the generated code.

    n: int
        length of the separator
    """
    txt = "."*n
    comment = '!'
    return '{0} {1}\n'.format(comment, txt)
# ...

class Compiler(object):
    """Base class for Code compiler for the Pyccel Grammar"""
    def __init__(self, codegen, compiler, \
                 flags=None, accelerator=None, \
                 binary=None, debug=False, \
                 inline=False, include=[], libdir=[], libs=[]):
        """
        Constructor of the code compiler.

        codegen: pyccel.codegen.Codegen
            a generation code object.
        compiler: str
            used compiler for the target language.
        flags: str
            compiler flags
        accelerator: str
            name of the selected accelerator.
            For the moment, only 'openmp' is available
        binary: str
            name of the binary file to generate.
        debug: bool
            add some useful prints that may help for debugging.
        inline: bool
            set to True, if the file is being load inside a python session.
        include: list
            list of include directories paths
        libdir: list
            list of lib directories paths
        libs: list
            list of libraries to link with
        """
        self._codegen     = codegen
        self._compiler    = compiler
        self._binary      = binary
        self._debug       = debug
        self._inline      = inline
        self._accelerator = accelerator
        self._include     = include
        self._libdir      = libdir
        self._libs        = libs

        if flags:
            self._flags = flags
        else:
            self._flags = self.construct_flags()

    @property
    def codegen(self):
        """Returns the used codegen"""
        return self._codegen

    @property
    def compiler(self):
        """Returns the used compiler"""
        return self._compiler

    @property
    def flags(self):
        """Returns the used flags"""
        return self._flags

    @property
    def binary(self):
        """Returns the used binary"""
        return self._binary

    @property
    def debug(self):
        """Returns True if in debug mode"""
        return self._debug

    @property
    def inline(self):
        """Returns True if in inline mode"""
        return self._inline

    @property
    def accelerator(self):
        """Returns the used accelerator"""
        return self._accelerator

    @property
    def include(self):
        """Returns include paths"""
        return self._include

    @property
    def libdir(self):
        """Returns lib paths"""
        return self._libdir

    @property
    def libs(self):
        """Returns libraries to link with"""
        return self._libs

    def construct_flags(self):
        """
        Constructs compiling flags
        """
        # TODO use constructor and a dict to map flags w.r.t the compiler
        _avail_compilers = ['gfortran', 'mpif90', 'pgfortran']

        compiler    = self.compiler
        debug       = self.debug
        inline      = self.inline
        accelerator = self.accelerator
        include     = self.include
        libdir      = self.libdir

        if not(compiler in _avail_compilers):
            raise ValueError("Only {0} are available.".format(_avail_compilers))

        flags = " -O2 "
        if compiler == "gfortran":
            if debug:
                flags += " -fbounds-check "

        if compiler == "mpif90":
            if debug:
                flags += " -fbounds-check "

        if not (accelerator is None):
            if accelerator == "openmp":
                flags += " -fopenmp "
            else:
                raise ValueError("Only openmp is available")

        if isinstance(include, str):
            include = [include]
        if len(include) > 0:
            flags += ' '.join(' -I{0}'.format(i) for i in include)

        if isinstance(libdir, str):
            libdir = [libdir]
        if len(libdir) > 0:
            flags += ' '.join(' -L{0}'.format(i) for i in libdir)

        return flags

    def compile(self, verbose=False):
        """
        Compiles the generated file.

        verbose: bool
            talk more
        """
        compiler  = self.compiler
        flags     = self.flags
        inline    = self.inline
        libs      = self.libs
        filename  = self.codegen.filename_out
        is_module = self.codegen.is_module
        modules   = self.codegen.modules

        ignored_modules  = ['plaf', 'spl', 'disco', 'fema']
        ignored_modules += ['plf', 'dsc', 'jrk']
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

        modules = [m for m in modules if not _ignore_module(m)]

        binary = ""
        if self.binary is None:
            if not is_module:
                binary = filename.split('.')[0]
        else:
            binary = self.binary

        o_code = ''
        if not inline:
            if not is_module:
                o_code = "-o"
            else:
                flags += ' -c '
        else:
            o_code = '-m'
            flags  = '--quiet -c'
            binary = 'external'
            # TODO improve
            compiler = 'f2py'

        m_code = ' '.join('{}.o '.format(m) for m in modules)

        if isinstance(libs, str):
            libs = [libs]
        if len(libs) > 0:
            libs = ' '.join(' -l{0}'.format(i) for i in libs)
        else:
            libs = ''

        cmd = '{0} {1} {2} {3} {4} {5} {6}'.format( \
            compiler, flags, m_code, filename, o_code, binary, libs)

        if verbose:
            print cmd

        os.system(cmd)

        self._binary = binary
# ...

# ...
def build_file(filename, language, compiler, \
               execute=False, accelerator=None, \
               debug=False, verbose=False, show=False, \
               inline=False, name=None, \
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

    # ...
    from pyccel.imports.utilities import find_imports

    d = find_imports(filename=filename)

    imports = {}

    ignored_modules.append('pyccel')
    for n in pyccel_modules:
        ignored_modules.append('pyccel.{0}'.format(n))

    # TODO remove. for the moment we use 'from spl.bspline import *'
    ignored_modules.append('plaf')
    ignored_modules.append('plf')
    ignored_modules.append('spl')
    ignored_modules.append('disco')
    ignored_modules.append('dsc')
    ignored_modules.append('fema')
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

    for key, value in d.items():
        if not _ignore_module(key):
            imports[key] = value

    ms = []
    for module, names in imports.items():
        codegen_m = FCodegen(filename=module+".py", name=module, is_module=True)
        codegen_m.doprint(language=language, accelerator=accelerator, \
                          ignored_modules=ignored_modules, \
                          with_mpi=with_mpi)
        ms.append(codegen_m)

    codegen = FCodegen(filename=filename, name=name)
    s=codegen.doprint(language=language, accelerator=accelerator, \
                     ignored_modules=ignored_modules, with_mpi=with_mpi, \
                     pyccel_modules=pyccel_modules)
    if show:
        print('========Fortran_Code========')
        print(s)
        print('============================')
        print ">>> Codegen :", name, " done."

    modules = codegen.modules
    # ...

    # ...
    if single_file:
        pyccel_stmts  = codegen.ast.extra_stmts
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
