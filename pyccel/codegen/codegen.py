# coding: utf-8

import os

from sympy.core import Tuple

from pyccel.codegen.printing import fcode

from pyccel.ast.core import subs
from pyccel.ast.core import is_valid_module
from pyccel.ast.core import DataType, DataTypeFactory
from pyccel.ast.core import (Range, Tensor, Block, ParallelBlock, \
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

from pyccel.ast.parallel.mpi import MPI_Tensor
from pyccel.ast.parallel.mpi import mpify
from pyccel.ast.parallel.openmp import openmpfy

from pyccel.parser.parser  import PyccelParser
from pyccel.parser.syntax.openmp import OpenmpStmt

from pyccel.parser.utilities import find_imports


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
def make_tmp_file(filename, output_dir=None):
    """
    returns a temporary file of extension .pyccel that will be decorated with
    indent/dedent so that textX can find the blocks easily.

    filename: str
        name of the file to parse.

    output_dir: str
        directory to store pyccel file
    """
    name = filename.split('.py')[0]
    if output_dir:
        name = os.path.basename(name)
        name = os.path.join(output_dir, name)
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
                 output_dir=None, \
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
        output_dir: str
            output directory to store pyccel files and generated files
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

        filename_tmp = make_tmp_file(filename, output_dir=output_dir)
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
        self._output_dir   = output_dir

    @property
    def filename(self):
        """Returns the name of the file to convert."""
        return self._filename

    @property
    def filename_out(self):
        """Returns the name of the output file."""
        return self._filename_out

    @property
    def output_dir(self):
        """Returns the output directory."""
        return self._output_dir

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
                pyccel_modules=[], \
                user_modules=[]):
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
        user_modules: list
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

        # ...
        with_openmp = False
        if accelerator:
            if accelerator == "openmp":
                with_openmp = True
            else:
                raise ValueError("Only openmp is available")
        # ...

        # ... TODO use Import class
        if with_mpi:
            imports += "use MPI\n"
        # ...

        # ... TODO use Import class
        if with_openmp:
            imports += "use omp_lib\n"
        # ...

        # ...
        pyccel = PyccelParser()
        ast    = pyccel.parse_from_file(filename)
        # ...

        # ... TODO improve
        for i in user_modules:
            print ('>>>>>>>>>>>>>> ', i)
            imports += "use {0}\n".format(i)
        # ...

        # ...
        stmts = ast.expr
        nm = ast.get_namespace()
        print nm
        # ...

        # ...
        if with_mpi:
            stmts = [mpify(s) for s in stmts]
        # ...

        # ...
        if with_openmp:
            stmts = [openmpfy(s) for s in stmts]
        # ...

        for stmt in stmts:
#            print stmt
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
            elif isinstance(stmt, ParallelBlock):
                body += printer(stmt) + "\n"
                # TODO use local vars to a parallel block (offloading?)
#                for dec in stmt.declarations:
#                    preludes += "\n" + printer(dec) + "\n"
            elif isinstance(stmt, Block):
                # TODO - now we can apply printer to Block directly. must be
                #      updated here => remove the body loop
                #      - printer(stmt) must return only body code, then preludes
                #      are treated somewhere?
                for s in stmt.body:
                    body += "\n" + printer(s) + "\n"
                for dec in stmt.declarations:
                    preludes += "\n" + printer(dec) + "\n"
            elif isinstance(stmt, Module):
                body += "\n" + printer(stmt) + "\n"
            else:
                if True:
                    print(("> uncovered statement of type : ", type(stmt)))
                else:
                    raise Exception('Statement not yet handled.')
        # ...

#        import sys; sys.exit(0)

        # ...
        for key, dec in list(ast.declarations.items()):
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
            is_module = is_valid_module(stmts)
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
