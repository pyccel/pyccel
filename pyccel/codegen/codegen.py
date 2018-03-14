# coding: utf-8

from pyccel.parser import Parser
import os

from pyccel.codegen.printing import fcode

from pyccel.ast.core import FunctionDef, ClassDef, Module, Program, Import
from pyccel.ast.core import Header, EmptyLine, Comment
from pyccel.ast.core import Assign, AliasAssign, SymbolicAssign
from pyccel.ast.core import Variable,DottedName
from pyccel.ast.core import For
from pyccel.ast.core import Is

from pyccel.parser.errors import Errors, PyccelCodegenError
# TODO improve this import
from pyccel.parser.messages import *

_extension_registry = {'fortran': 'f90'}

class Codegen(object):
    """Abstract class for code generator."""
    def __init__(self, expr, name):
        """Constructor for Codegen.

        expr: sympy expression
            expression representing the AST as a sympy object

        name: str
            name of the generated module or program.
        """
        self._ast = expr
        self._name = name
        self._kind = None
        self._code = None
        self._language = None

        self._stmts = {}
        _structs = ['imports', 'body', 'routines', 'classes', 'modules',
                    'variables']
        for key in _structs:
            self._stmts[key] = []

        self._collect_statments()
        self._set_kind()

    @property
    def name(self):
        """Returns the name associated to the source code"""
        return self._name

    @property
    def kind(self):
        """Returns the kind of the source code: Module, Program or None."""
        return self._kind

    @property
    def imports(self):
        """Returns the imports of the source code."""
        return self._stmts['imports']

    @property
    def variables(self):
        """Returns the variables of the source code."""
        return self._stmts['variables']

    @property
    def body(self):
        """Returns the body of the source code, if it is a Program or Module."""
        return self._stmts['body']

    @property
    def routines(self):
        """Returns functions/subroutines."""
        return self._stmts['routines']

    @property
    def classes(self):
        """Returns the classes if Module."""
        return self._stmts['classes']

    @property
    def modules(self):
        """Returns the modules if Program."""
        return self._stmts['modules']

    @property
    def is_module(self):
        """Returns True if a Module."""
        return self.kind == 'module'

    @property
    def is_program(self):
        """Returns True if a Program."""
        return self.kind == 'program'

    @property
    def ast(self):
        """Returns the AST."""
        return self._ast

    @property
    def expr(self):
        """Returns the AST after Module/Program treatment."""
        return self._expr

    @property
    def language(self):
        """Returns the used language"""
        return self._language

    @property
    def code(self):
        """Returns the generated code."""
        return self._code

    def _collect_statments(self):
        """Collects statments and split them into routines, classes, etc."""

        errors = Errors()
        errors.set_parser_stage('codegen')

        variables = []
        routines = []
        classes = []
        imports = []
        modules = []
        body = []
        decs = []

        for stmt in self.ast:
            if isinstance(stmt, FunctionDef):
                routines += [stmt]
            elif isinstance(stmt, ClassDef):
                classes += [stmt]
            elif isinstance(stmt, Import):
                imports += [stmt]
            elif isinstance(stmt, Module):
                modules += [stmt]
            else:
                # TODO improve later, as in the old codegen
                # we don't generate code for symbolic assignments
                if isinstance(stmt, SymbolicAssign):
                    errors.report(FOUND_SYMBOLIC_ASSIGN, symbol=stmt.lhs, severity='warning')
                    body += [Comment(str(stmt))]
                elif isinstance(stmt, Assign) and isinstance(stmt.rhs, Is):
                    errors.report(FOUND_IS_IN_ASSIGN, symbol=stmt.lhs, severity='warning')
                    body += [Comment(str(stmt))]
                else:
                    body += [stmt]

                if isinstance(stmt, (Assign, AliasAssign)):
                    if isinstance(stmt.lhs, Variable):
                        if not isinstance(stmt.lhs.name,DottedName):
                            variables += [stmt.lhs]
                            #we only add the variables which are not DottedName
                if isinstance(stmt, For):
                    if isinstance(stmt.target, Variable):
                        variables += [stmt.target]

        self._stmts['imports'] = imports
        self._stmts['variables'] = variables
        self._stmts['body'] = body
        self._stmts['routines'] = routines
        self._stmts['classes'] = classes
        self._stmts['modules'] = modules

        errors.check()


    # TODO improve to have a kind = None
    #      => must have a condition to be aprogram
    def _set_kind(self):
        """Finds the source code kind."""
        # ...
        _stmts = (Header, EmptyLine, Comment)
        body = self.body

        ls = [i for i in body if not isinstance(i, _stmts)]
        is_module = (len(ls) == 0)
        if is_module:
            self._kind = 'module'
        else:
            self._kind = 'program'
        # ...

        # ...
        expr = None
        if self.is_module:
            expr = Module(self.name,
                          self.variables,
                          self.routines,
                          self.classes,
                          imports=self.imports)
        elif self.is_program:
            expr = Program(self.name,
                           self.variables,
                           self.routines,
                           self.classes,
                           self.body,
                           imports=self.imports,
                           modules=self.modules)
        else:
            raise NotImplementedError('TODO')

        self._expr = expr
        # ...

    def doprint(self, **settings):
        """Prints the code in the target language."""
        # ... finds the target language
        language = settings.pop('language', 'fortran')
        if not(language == 'fortran'):
            raise ValueError('Only fortran is available')
        self._language = language
        # ...

        # ... define the printing function to be used
        printer = settings.pop('printer', fcode)
        # ...

        # ...
        code = printer(self.expr)
        # ...

        self._code = code

        return code

    def export(self, filename=None):
        ext = _extension_registry[self.language]
        if filename is None:
            filename = '{name}.{ext}'.format(name=self.name, ext=ext)
        else:
            raise NotImplementedError('TODO')

        code = self.code
        f = open(filename, "w")
        for line in code:
            f.write(line)
        f.close()

class FCodegen(Codegen):
    pass
