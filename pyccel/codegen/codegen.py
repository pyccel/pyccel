# coding: utf-8

import os

import re

from sympy.core import Tuple
from sympy import Lambda

from pyccel.codegen.printing import fcode

from pyccel.parser.syntax.core import get_headers
from pyccel.parser.syntax.core import get_namespace
from pyccel.parser.syntax.core import clean_namespace

from pyccel.ast.core import subs
from pyccel.ast.core import Header
from pyccel.ast.core import is_valid_module
from pyccel.ast.core import EmptyLine
from pyccel.ast.core import DataType, DataTypeFactory
from pyccel.ast.core import (Range, Tensor, Block, ParallelBlock,
                             For, Assign, Declare, Variable,
                             Load, Eval,
                             NativeRange, NativeTensor,
                             FunctionHeader, ClassHeader,
                             MethodHeader, VariableHeader,
                             datatype, While, With, NativeFloat,
                             EqualityStmt, NotequalStmt,
                             AugAssign, FunctionCall,
                             FunctionDef,MethodCall, ClassDef,
                             Del, Print, Import, Comment, AnnotatedComment,
                             IndexedVariable, Slice, Assert, If,
                             Vector, Stencil, Ceil, Break,
                             Zeros, Ones, Array, ZerosLike, Shape, Len,
                             Dot, Sign, IndexedElement, DottedName,
                             Module, Program)

from pyccel.ast.parallel.mpi     import mpify
from pyccel.ast.parallel.openmp  import ompfy
from pyccel.ast.parallel.openacc import accfy

from pyccel.parser.parser    import PyccelParser
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
def updateNewLineInList(s, kind='list'):
    """
    removing 'new line', inside a list, tuple, dict
    """
    if kind == 'list':
#        rule = '\[(.*[\w\W\s\S]+.*)\]'
#        rule = r'\[(.*\n[\w\W\s\S]+.*)\]'
#        rule = '\[([\w,=:]*\n[\w\W\s\S]+.*)\]'
#        rule = '\[([\w,=: \t]*\n[\w\W\s\S]+.*)\]'

        rule = '\[([\w,=: \t]*\n[^)]+)\]'

        leftLim = '['
        rightLim = ']'
    elif kind == 'tuple':
#        rule = '\((.*[\w\W\s\S]+.*)\)'
#        rule = r'\((.*\n[\w\W\s\S]+.*)\)'
#        rule = '\(([\w,=:]*\n[\w\W\s\S]+.*)\)'
#        rule = '\(([\w,=: \t]*\n[\w\W\s\S]+.*)\)'

        rule = '\(([\w,=: \t]*\n[^)]+)\)'

        leftLim = '('
        rightLim = ')'
    else:
        raise ValueError('Expecting list or tuple values for kind')

    p = re.compile(rule)
    #p = re.compile(rule, re.IGNORECASE)

    # ...
    list_data = p.split(s) # split the whole text with respect to the rule
    list_exp  = p.findall(s) # get all expressions to replace

    if len(list_exp) == 0:
        return s

    _format = lambda s: s.replace('\n', ' ')

    list_text = ""
    for data in list_data:
        # TODO improve this in case of \t, etc
        #      this will be needed only if we want
        #      to have \n inside expressions.
        new_data = data
        if data in list_exp:
            new_data = leftLim + _format(data) + rightLim
        list_text += new_data

    return list_text
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
        is_annotated = (line.lstrip()[0:2] == '#$')
        skipped = is_empty or is_comment

        if n == depth * tab + tab:
            depth += 1
            lines_new += "indent\n" + "\n"
            lines_new += line
        elif not skipped:
            d = n // tab
            if (d > 0) or (n==0):
                old = delta(old_line)
                m = (old - n) // tab
                depth -= m
                for j in range(0, m):
                    lines_new += "dedent\n" + "\n"
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
#    # ...
#    f = open(filename)
#    lines = f.readlines()
#    f.close()
#    # ...

    # ...
    f = open(filename)
    code = f.read()
    f.close()

    # remove new lines between '(' and ')'
    code = updateNewLineInList(code, kind='tuple')
    code = updateNewLineInList(code, kind='list')
    lines = code.split('\n')
    lines = [l+'\n' for l in lines]
    # ...

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
                 modules=None):
        """Constructor for the Codegen class.

        filename: str
            name of the file to parse.
        name: str
            name of the generated module or program.
            if not given, 'main' will be used in the case of a program, and
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
        """
        # ... TODO improve once TextX will handle indentation
        clean(filename)

        # check if the file is a header
        is_header = (filename.split('.')[-1] == 'pyh')

        filename_tmp = make_tmp_file(filename, output_dir=output_dir)
        preprocess(filename, filename_tmp)

        # file extension is changed to .pyccel
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
        self._body         = body
        self._preludes     = preludes
        self._routines     = routines
        self._classes      = classes
        self._modules      = modules
        self._printer      = None
        self._output_dir   = output_dir
        self._namespace    = None
        self._headers      = None
        self._is_header    = is_header

        # ...
        pyccel = PyccelParser()
        ast    = pyccel.parse_from_file(filename)

        self._ast = ast
        # ...

        # ... this is important to update the namespace
        # old namespace
        namespace_user = get_namespace()

        stmts = ast.expr

        # new namespace
        self._namespace = get_namespace()
        self._headers   = get_headers()
        # ...

        # ...................................................
        # reorganizing the code to get a Module or a Program
        # ...................................................
        variables = []
        funcs     = []
        classes   = []
        imports   = []
        modules   = []
        body      = []
        decs      = []

        # ... TODO update Declare so that we don't use a list of variables
        for key, dec in list(ast.declarations.items()):
            decs += [dec]
            variables += [dec.variables[0]]
        # ...

        # ...
        for stmt in stmts:
            if isinstance(stmt, FunctionDef):
                funcs += [stmt]
            elif isinstance(stmt, ClassDef):
                classes += [stmt]
            elif isinstance(stmt, Import):
                imports += [stmt]
            else:
                body += [stmt]
        # ...

        # ...
        _stmts = (Header, EmptyLine, Comment)

        ls = [i for i in body if not isinstance(i, _stmts)]
        is_module = (len(ls) == 0)
        # ...

        # ...
        expr = None
        if is_module:
            expr = Module(name, variables, funcs, classes,
                          imports=imports)
        else:
            expr = Program(name, variables, funcs, classes, body,
                           imports=imports, modules=modules)
        self._expr = expr
        # ...
        # ...................................................

        # ... cleaning
        clean_namespace()
        # ...

    @property
    def expr(self):
        """Returns the ast as sympy object: Module or Program."""
        return self._expr

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
        return isinstance(self.expr, Module)

    @property
    def printer(self):
        """Returns the used printer"""
        return self._printer

    @property
    def namespace(self):
        """Returns the namespace."""
        if not self._is_header:
            return self._namespace

        # in the case of a header file, we need to convert the headers to
        # FunctionDef or ClassCef
        namespace = self._namespace

        # TODO must create the definition only if only the header exists
        #      for the moment, we only export functions
        if self.is_header:
            for k,v in self.headers.items():
                if isinstance(v, FunctionHeader) and not isinstance(v, MethodHeader):
                    f = v.create_definition()
                    namespace[k] = f

        return namespace

    @property
    def headers(self):
        """Returns the headers."""
        return self._headers

    @property
    def metavars(self):
        """Returns meta-variables."""
        return self._metavars

    @property
    def is_header(self):
        """Returns True if generated code is a header"""
        return self._is_header

    def doprint(self, language,
                accelerator=None,
                ignored_modules=[],
                with_mpi=False,
                pyccel_modules=[],
                user_modules=[],
                enable_metavars=True):
        """Generate code for a given language.

        metavariables that starts with __ will be appended into a dictionary
        with their corresponding value.
        Their declarations will be ignored when printing the code.

        language: str
            target language. Possible values {"fortran"}
        accelerator: str
            name of the selected accelerator.
            One among ('openmp', 'openacc')
        ignored_modules: list
            list of modules to ignore.
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
        metavars = {}

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
        with_openmp  = False
        with_openacc = False
        if accelerator:
            if accelerator == "openmp":
                with_openmp = True
            elif accelerator == "openacc":
                with_openacc = True
            else:
                raise ValueError("Only openmp is available")
        # ...

        ####################################################

        # ... TODO improve
        for i in user_modules:
            imports += "use {0}\n".format(i)
        # ...

        # ...
        expr = self.expr
        # ...

        # ...
        settings = {}

        settings['ignored_modules'] = ignored_modules
        settings['with_openmp']     = with_openmp
        settings['with_openacc']    = with_openacc
        settings['with_mpi']        = with_mpi

        e, info = pyccelize(expr, **settings)
        metavars = info['metavars']
        code = printer(e)
        # ...

        # ...
        self._imports   = imports
        self._preludes  = preludes
        self._body      = body
        self._routines  = routines
        self._classes   = classes
        self._modules   = list(set(modules))
        self._language  = language
        self._metavars  = metavars
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
def pyccelize(expr, **settings):
    """."""

    # ...
    with_mpi  = False
    if 'with_mpi' in settings:
        with_mpi  = settings['with_mpi']
    # ...

    # ...
    with_openmp  = False
    if 'with_openmp' in settings:
        with_openmp  = settings['with_openmp']
    # ...

    # ...
    with_openacc  = False
    if 'with_openacc' in settings:
        with_openacc  = settings['with_openacc']
    # ...

    # ...
    ignored_modules  = []
    if 'ignored_modules' in settings:
        ignored_modules  = settings['ignored_modules']
    # ...

    # ...
    if not isinstance(expr, (Module, Program)):
        raise NotImplementedError('')
    # ...

    # ...
    if isinstance(expr, (Module, Program)):
        # ...
        name      = expr.name
        variables = expr.variables
        funcs     = expr.funcs
        classes   = expr.classes

        imports  = []
        modules  = []
        body     = []
        decs     = []
        metavars = {}
        info     = {}
        # ...

        # ...
        for stmt in expr.imports:
            name = str(stmt.fil)
            if not(name in ignored_modules):
                imports += [stmt]
        # ...

#        # ...
#        def _construct_prelude(stmt):
#            preludes = ''
#            if isinstance(stmt, (list, tuple, Tuple)):
#                for dec in stmt:
#                    preludes += _construct_prelude(dec)
#            if isinstance(stmt, (For, While)):
#                preludes += _construct_prelude(stmt.body)
#            if isinstance(stmt, Block):
#                for dec in stmt.declarations:
#                    preludes += printer(dec) + "\n"
#            return preludes
#
#        preludes += _construct_prelude(expr.body)
#        # ...

        # ...
        _hidden_stmts = (Eval, Load)
        if isinstance(expr, Program):
            for stmt in expr.body:
                # Variable is also ignored, since we can export them in headers
                if isinstance(stmt, (Variable, Lambda)):
                    continue
                elif isinstance(stmt, Assign):
                    if isinstance(stmt.rhs, (Range, Tensor)):
                        continue
                    elif isinstance(stmt.lhs, Variable):
                        if (isinstance(stmt.lhs.name, str) and
                            stmt.lhs.name.startswith('__')):
                            metavars[stmt.lhs.name] = stmt.rhs
                        else:
                            body += [stmt]
                    else:
                        body += [stmt]
                elif isinstance(stmt, _hidden_stmts):
                    continue
                elif isinstance(stmt, Block):
                    # TODO - now we can apply printer to Block directly. must be
                    #      updated here => remove the body loop
                    #      - printer(stmt) must return only body code, then preludes
                    #      are treated somewhere?
                    for s in stmt.body:
                        body += [stmt]
#                    for dec in stmt.declarations:
#                        preludes += "\n" + printer(dec) + "\n"
                else:
                    body += [stmt]
        # ...

        # ...
        if isinstance(expr, Module):
            expr = Module(name, variables, funcs, classes,
                          imports=imports)
        else:
            expr = Program(name, variables, funcs, classes, body,
                           imports=imports, modules=modules)
        # ...

        info['metavars'] = metavars
    # ...

    # ...
    if with_mpi:
        expr = mpify(expr)
    # ...

    # ...
    if with_openmp:
        expr = ompfy(expr)
    # ...

    # ...
    if with_openacc:
        expr = accfy(expr)
    # ...

    return expr, info
# ...
