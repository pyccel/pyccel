# coding: utf-8


from __future__ import print_function, division

import os
import textwrap
from sympy import sympify
from sympy.core import Symbol, S, Tuple, Equality, Function, Basic
from sympy.matrices import (MatrixSymbol, ImmutableMatrix, MatrixBase,
                            MatrixExpr, MatrixSlice)
from sympy.tensor import Idx, Indexed, IndexedBase
from sympy import Eq
from sympy.core.compatibility import is_sequence, StringIO, string_types

from pyccel.types.ast import Assign, For
from pyccel.printers.fcode import fcode, FCodePrinter
from pyccel.printers.ccode import ccode, CCodePrinter
from pyccel.printers.luacode import lua_code, LuaCodePrinter


__all__ = ["codegen"]

class DataType(object):
    """Holds strings for a certain datatype in different languages."""
    def __init__(self, fname, luaname):
        self.fname   = fname
        self.luaname = luaname


default_datatypes = {
    "int": DataType("INTEGER*4", ""),
    "float": DataType("REAL*8", ""),
}


def get_default_datatype(expr):
    """Derives an appropriate datatype based on the expression."""
    if expr.is_integer:
        return default_datatypes["int"]
    elif isinstance(expr, MatrixBase):
        for element in expr:
            if not element.is_integer:
                return default_datatypes["float"]
        return default_datatypes["int"]
    else:
        return default_datatypes["float"]


class Variable(object):
    """Represents a typed variable."""

    def __init__(self, name, datatype=None, dimensions=None, precision=None):
        """Return a new variable.

        Parameters
        ==========

        name : Symbol or MatrixSymbol

        datatype : optional
            When not given, the data type will be guessed based on the
            assumptions on the symbol argument.

        dimension : sequence containing tupes, optional
            If present, the argument is interpreted as an array, where this
            sequence of tuples specifies (lower, upper) bounds for each
            index of the array.

        precision : int, optional
            Controls the precision of floating point constants.

        """
        if not isinstance(name, (Symbol, MatrixSymbol)):
            raise TypeError("The first argument must be a sympy symbol.")
        if datatype is None:
            datatype = get_default_datatype(name)
        elif not isinstance(datatype, DataType):
            raise TypeError("The (optional) `datatype' argument must be an "
                            "instance of the DataType class.")
        if dimensions and not isinstance(dimensions, (tuple, list)):
            raise TypeError(
                "The dimension argument must be a sequence of tuples")

        self._name = name
        self._datatype = {
            'LUA': datatype.luaname,
            'FORTRAN': datatype.fname,
        }
        self.dimensions = dimensions
        self.precision = precision

    def __str__(self):
        return "%s(%r)" % (self.__class__.__name__, self.name)

    __repr__ = __str__

    @property
    def name(self):
        return self._name

    def get_datatype(self, language):
        """Returns the datatype string for the requested language.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.utilities.codegen import Variable
        >>> x = Variable(Symbol('x'))
        >>> x.get_datatype('fortran')
        'REAL*8'

        """
        try:
            return self._datatype[language.upper()]
        except:
            print ("Has datatypes for languages: %s" % ", ".join(self._datatype))
            raise()

class Argument(Variable):
    """An abstract Argument data structure: a name and a data type.

    This structure is refined in the descendants below.

    """
    pass


class InputArgument(Argument):
    pass



class ResultBase(object):
    """Base class for all "outgoing" information from a routine.

    Objects of this class stores a sympy expression, and a sympy object
    representing a result variable that will be used in the generated code
    only if necessary.

    """
    def __init__(self, expr, result_var):
        self.expr = expr
        self.result_var = result_var

    def __str__(self):
        return "%s(%r, %r)" % (self.__class__.__name__, self.expr,
            self.result_var)

    __repr__ = __str__


class OutputArgument(Argument, ResultBase):
    """OutputArgument are always initialized in the routine."""

    def __init__(self, name, result_var, expr, datatype=None, dimensions=None, precision=None):
        """Return a new variable.

        Parameters
        ==========

        name : Symbol, MatrixSymbol
            The name of this variable.  When used for code generation, this
            might appear, for example, in the prototype of function in the
            argument list.

        result_var : Symbol, Indexed
            Something that can be used to assign a value to this variable.
            Typically the same as `name` but for Indexed this should be e.g.,
            "y[i]" whereas `name` should be the Symbol "y".

        expr : object
            The expression that should be output, typically a SymPy
            expression.

        datatype : optional
            When not given, the data type will be guessed based on the
            assumptions on the symbol argument.

        dimension : sequence containing tupes, optional
            If present, the argument is interpreted as an array, where this
            sequence of tuples specifies (lower, upper) bounds for each
            index of the array.

        precision : int, optional
            Controls the precision of floating point constants.

        """

        Argument.__init__(self, name, datatype, dimensions, precision)
        ResultBase.__init__(self, expr, result_var)

    def __str__(self):
        return "%s(%r, %r, %r)" % (self.__class__.__name__, self.name, self.expr,
            self.result_var)

    __repr__ = __str__


class InOutArgument(Argument, ResultBase):
    """InOutArgument are never initialized in the routine."""

    def __init__(self, name, result_var, expr, datatype=None, dimensions=None, precision=None):
        if not datatype:
            datatype = get_default_datatype(expr)
        Argument.__init__(self, name, datatype, dimensions, precision)
        ResultBase.__init__(self, expr, result_var)
    __init__.__doc__ = OutputArgument.__init__.__doc__


    def __str__(self):
        return "%s(%r, %r, %r)" % (self.__class__.__name__, self.name, self.expr,
            self.result_var)

    __repr__ = __str__


class Result(Variable, ResultBase):
    """An expression for a return value.

    The name result is used to avoid conflicts with the reserved word
    "return" in the python language.  It is also shorter than ReturnValue.

    These may or may not need a name in the destination (e.g., "return(x*y)"
    might return a value without ever naming it).

    """

    def __init__(self, expr, name=None, result_var=None, datatype=None,
                 dimensions=None, precision=None):
        """Initialize a return value.

        Parameters
        ==========

        expr : SymPy expression

        name : Symbol, MatrixSymbol, optional
            The name of this return variable.  When used for code generation,
            this might appear, for example, in the prototype of function in a
            list of return values.  A dummy name is generated if omitted.

        result_var : Symbol, Indexed, optional
            Something that can be used to assign a value to this variable.
            Typically the same as `name` but for Indexed this should be e.g.,
            "y[i]" whereas `name` should be the Symbol "y".  Defaults to
            `name` if omitted.

        datatype : optional
            When not given, the data type will be guessed based on the
            assumptions on the symbol argument.

        dimension : sequence containing tupes, optional
            If present, this variable is interpreted as an array,
            where this sequence of tuples specifies (lower, upper)
            bounds for each index of the array.

        precision : int, optional
            Controls the precision of floating point constants.

        """
        # Basic because it is the base class for all types of expressions
        if not isinstance(expr, (Basic, MatrixBase)):
            raise TypeError("The first argument must be a sympy expression.")

        if name is None:
            name = 'result_%d' % abs(hash(expr))

        if isinstance(name, string_types):
            if isinstance(expr, (MatrixBase, MatrixExpr)):
                name = MatrixSymbol(name, *expr.shape)
            else:
                name = Symbol(name)

        if result_var is None:
            result_var = name

        Variable.__init__(self, name, datatype=datatype,
                          dimensions=dimensions, precision=precision)
        ResultBase.__init__(self, expr, result_var)


class Routine(object):
    """Generic description of evaluation routine for set of expressions.

    A CodeGen class can translate instances of this class into code in a
    particular language.  The routine specification covers all the features
    present in these languages.  The CodeGen part must raise an exception
    when certain features are not present in the target language.  For
    example, multiple return values are possible in Python, but not in C or
    Fortran.  Another example: Fortran and Python support complex numbers,
    while C does not.

    """

    def __init__(self, name, arguments, results, statements, local_vars, global_vars):
        """Initialize a Routine instance.

        Parameters
        ==========

        name : string
            Name of the routine.

        arguments : list of Arguments
            These are things that appear in arguments of a routine, often
            appearing on the right-hand side of a function call.  These are
            commonly InputArguments but in some languages, they can also be
            OutputArguments or InOutArguments (e.g., pass-by-reference in C
            code).

        results : list of Results
            These are the return values of the routine, often appearing on
            the left-hand side of a function call.  The difference between
            Results and OutputArguments and when you should use each is
            language-specific.

        statements: list of Statements
            Anything that is does not appear in results

        local_vars : list of Symbols
            These are used internally by the routine.

        global_vars : list of Symbols
            Variables which will not be passed into the function.

        """

        # extract all input symbols and all symbols appearing in an expression
        input_symbols = set([])
        symbols = set([])
        for arg in arguments:
            if isinstance(arg, OutputArgument):
                symbols.update(arg.expr.free_symbols)
            elif isinstance(arg, InputArgument):
                input_symbols.add(arg.name)
            elif isinstance(arg, InOutArgument):
                input_symbols.add(arg.name)
                symbols.update(arg.expr.free_symbols)
            else:
                raise ValueError("Unknown Routine argument: %s" % arg)

        for r in results:
            if not isinstance(r, Result):
                raise ValueError("Unknown Routine result: %s" % r)

            try:
                symbols.update(r.expr.free_symbols)
            except:
                pass

            try:
                symbols.update(r.expr)
            except:
                pass

#            if (not isinstance(r, Assign)) and (not isinstance(r, For)):
#                if not isinstance(r, Result):
#                    raise ValueError("Unknown Routine result: %s" % r)
#                symbols.update(r.expr.free_symbols)

        for stmt in statements:
            if not isinstance(stmt, (Assign, Equality, For)):
                raise ValueError("Unknown Routine statement: %s" % stmt)
#            symbols.update(stmt.expr.free_symbols)
#            symbols.update(stmt.expr.free_symbols - set(arguments))

        symbols = set([s.label if isinstance(s, Idx) else s for s in symbols])

        # Check that all symbols in the expressions are covered by
        # InputArguments/InOutArguments---subset because user could
        # specify additional (unused) InputArguments or local_vars.
        notcovered = symbols.difference(
            input_symbols.union(local_vars).union(global_vars))

        if notcovered != set([]):
            raise ValueError("Symbols needed for output are not in input or local " +
                             ", ".join([str(x) for x in notcovered]))

        self.name = name
        self.arguments = arguments
        self.results = results
        self.statements = statements
        self.local_vars = local_vars
        self.global_vars = global_vars
        self.symbols = symbols

    def __str__(self):
        return self.__class__.__name__ + "({name!r}, {arguments}, {results}, {statements}, {local_vars}, {global_vars})".format(**self.__dict__)

    __repr__ = __str__

    @property
    def variables(self):
        """Returns a set of all variables possibly used in the routine.

        For routines with unnamed return values, the dummies that may or
        may not be used will be included in the set.

        """
        v = set([])
        for arg in self.arguments:
            v.add(arg.name)
        for res in self.results:
            v.add(res.result_var)
        return v

    @property
    def result_variables(self):
        """Returns a list of OutputArgument, InOutArgument and Result.

        If return values are present, they are at the end ot the list.
        """
        args = [arg for arg in self.arguments if isinstance(
            arg, (OutputArgument, InOutArgument))]
        args.extend(self.results)
        return args

    @property
    def stmt_variables(self):
        """Returns a set of all variables used for statements in the routine.

        """
        v = set([])
        for stmt in self.statements:
            if isinstance(stmt, Assign):
                v.add(stmt.lhs)
        return v


#
# Transformation of routine objects into code
#

class CodeGen(object):
    """Abstract class for the code generators."""

    def __init__(self, project="project"):
        """Initialize a code generator.

        Derived classes will offer more options that affect the generated
        code.

        """
        self.project = project

    def routine(self, name, expr, argument_sequence, statements, global_vars, \
                local_vars=None, return_vars=None):
        """Creates an Routine object that is appropriate for this language.

        This implementation is appropriate for at least C/Fortran.  Subclasses
        can override this if necessary.

        Here, we assume at most one return value (the l-value) which must be
        scalar.  Additional outputs are OutputArguments (e.g., pointers on
        right-hand-side or pass-by-reference).  Matrices are always returned
        via OutputArguments.  If ``argument_sequence`` is None, arguments will
        be ordered alphabetically, but with all InputArguments first, and then
        OutputArgument and InOutArguments.

        """
        if is_sequence(expr) and not isinstance(expr, (MatrixBase, MatrixExpr)):
            if not expr:
                raise ValueError("No expression given")
            expressions = Tuple(*expr)
        else:
            expressions = Tuple(expr)

        # local variables
        if local_vars is None:
            local_vars = set([])
        local_vars = local_vars.union({i.label for i in expressions.atoms(Idx)})

        # global variables
        global_vars = set() if global_vars is None else set(global_vars)

        # symbols that should be arguments
        free_symbols = set([])
        for expr in expressions:
            if not isinstance(expr, For):
                for f in expr.free_symbols:
                    free_symbols.add(f)
            else:
                free_symbols.add(expr.target)
                free_symbols.add(expr.iterable.stop)

        symbols = free_symbols - local_vars - global_vars
        new_symbols = set([])
        new_symbols.update(symbols)

        for symbol in symbols:
            if isinstance(symbol, Idx):
                new_symbols.remove(symbol)
                new_symbols.update(symbol.args[1].free_symbols)
        symbols = new_symbols

        # Decide whether to use output argument or return value
        return_val = []
        if not return_vars is None:
            return_val += list(return_vars)

        output_args = []
        stmts = []
        for expr in expressions:
            if isinstance(expr, Equality):
                out_arg = expr.lhs
                expr = expr.rhs
                if isinstance(out_arg, Indexed):
                    dims = tuple([ (S.Zero, dim - 1) for dim in out_arg.shape])
                    symbol = out_arg.base.label
                elif isinstance(out_arg, Symbol):
                    dims = []
                    symbol = out_arg
                elif isinstance(out_arg, MatrixSymbol):
                    dims = tuple([ (S.Zero, dim - 1) for dim in out_arg.shape])
                    symbol = out_arg
                else:
                    raise CodeGenError("Only Indexed, Symbol, or MatrixSymbol "
                                       "can define output arguments.")

                if expr.has(symbol):
                    output_args.append(
                        InOutArgument(symbol, out_arg, expr, dimensions=dims))
                elif not(symbol in local_vars):
                    output_args.append(
                        OutputArgument(symbol, out_arg, expr, dimensions=dims))

                # avoid duplicate arguments
                if not(symbol in local_vars):
                    symbols.remove(symbol)
            elif isinstance(expr, (ImmutableMatrix, MatrixSlice)):
                # Create a "dummy" MatrixSymbol to use as the Output arg
                out_arg = MatrixSymbol('out_%s' % abs(hash(expr)), *expr.shape)
                dims = tuple([(S.Zero, dim - 1) for dim in out_arg.shape])
                output_args.append(
                    OutputArgument(out_arg, out_arg, expr, dimensions=dims))
            elif isinstance(expr, Assign):
                out_arg = expr.lhs
                expr = expr.rhs
                if isinstance(out_arg, Indexed):
                    dims = tuple([ (S.Zero, dim - 1) for dim in out_arg.shape])
                    symbol = out_arg.base.label
                elif isinstance(out_arg, Symbol):
                    dims = []
                    symbol = out_arg
                elif isinstance(out_arg, MatrixSymbol):
                    dims = tuple([ (S.Zero, dim - 1) for dim in out_arg.shape])
                    symbol = out_arg
                else:
                    raise CodeGenError("Only Indexed, Symbol, or MatrixSymbol "
                                       "can define output arguments.")

                if (expr.has(symbol)) or (not(symbol in local_vars)):
                    output_args.append(
                        InOutArgument(symbol, out_arg, out_arg + S.Zero, dimensions=dims))

                # avoid duplicate arguments
                if not(symbol in local_vars):
                    symbols.remove(symbol)

                stmts.append(Assign(out_arg, expr))
            else:
                return_val.append(Result(expr))
#                if not isinstance(expr, Assign):
#                    return_val.append(Result(expr))


        # setup input argument list
        arg_list = []
        array_symbols = {}
        for array in expressions.atoms(Indexed):
            array_symbols[array.base.label] = array
        for array in expressions.atoms(MatrixSymbol):
            array_symbols[array] = array

        for symbol in sorted(symbols, key=str):
            if symbol in array_symbols:
                dims = []
                array = array_symbols[symbol]
                for dim in array.shape:
                    dims.append((S.Zero, dim - 1))
                metadata = {'dimensions': dims}
            else:
                metadata = {}

            arg_list.append(InputArgument(symbol, **metadata))

        output_args.sort(key=lambda x: str(x.name))
        arg_list.extend(output_args)

        if argument_sequence is not None:
            # if the user has supplied IndexedBase instances, we'll accept that
            new_sequence = []
            for arg in argument_sequence:
                if isinstance(arg, IndexedBase):
                    new_sequence.append(arg.label)
                else:
                    new_sequence.append(arg)
            argument_sequence = new_sequence

            missing = [x for x in arg_list if x.name not in argument_sequence]
            if missing:
                msg = "Argument list didn't specify: {0} "
                msg = msg.format(", ".join([str(m.name) for m in missing]))
                raise CodeGenArgumentListError(msg, missing)

            # create redundant arguments to produce the requested sequence
            name_arg_dict = {x.name: x for x in arg_list}

            new_args = []
            for symbol in argument_sequence:
                try:
                    new_args.append(name_arg_dict[symbol])
                except KeyError:
                    new_args.append(InputArgument(symbol))
            arg_list = new_args

        arg_list.sort(key=lambda x: str(x.name).lower())
        return Routine(name, arg_list, return_val, stmts, local_vars, global_vars)

    def write(self, routines, prefix, to_files=False, header=True, empty=True):
        """Writes all the source code files for the given routines.

        The generated source is returned as a list of (filename, contents)
        tuples, or is written to files (see below).  Each filename consists
        of the given prefix, appended with an appropriate extension.

        Parameters
        ==========

        routines : list
            A list of Routine instances to be written

        prefix : string
            The prefix for the output files

        to_files : bool, optional
            When True, the output is written to files.  Otherwise, a list
            of (filename, contents) tuples is returned.  [default: False]

        header : bool, optional
            When True, a header comment is included on top of each source
            file. [default: True]

        empty : bool, optional
            When True, empty lines are included to structure the source
            files. [default: True]

        """
        if to_files:
            for dump_fn in self.dump_fns:
                filename = "%s.%s" % (prefix, dump_fn.extension)
                with open(filename, "w") as f:
                    dump_fn(self, routines, f, prefix, header, empty)
        else:
            result = []
            for dump_fn in self.dump_fns:
                filename = "%s.%s" % (prefix, dump_fn.extension)
                contents = StringIO()
                dump_fn(self, routines, contents, prefix, header, empty)
                result.append((filename, contents.getvalue()))
            return result

    def dump_code(self, routines, f, prefix, header=True, empty=True):
        """Write the code by calling language specific methods.

        The generated file contains all the definitions of the routines in
        low-level code and refers to the header file if appropriate.

        Parameters
        ==========

        routines : list
            A list of Routine instances.

        f : file-like
            Where to write the file.

        prefix : string
            The filename prefix, used to refer to the proper header file.
            Only the basename of the prefix is used.

        header : bool, optional
            When True, a header comment is included on top of each source
            file.  [default : True]

        empty : bool, optional
            When True, empty lines are included to structure the source
            files.  [default : True]

        """

        code_lines = self._preprocessor_statements(prefix)

        for routine in routines:
            if empty:
                code_lines.append("\n")
            code_lines.extend(self._get_routine_opening(routine))
            code_lines.extend(self._declare_arguments(routine))
            code_lines.extend(self._declare_globals(routine))
            code_lines.extend(self._declare_locals(routine))
            if empty:
                code_lines.append("\n")
            code_lines.extend(self._call_printer(routine))
            if empty:
                code_lines.append("\n")
            code_lines.extend(self._get_routine_ending(routine))

        code_lines = self._indent_code(''.join(code_lines))

        if header:
            code_lines = ''.join(self._get_header() + [code_lines])

        if code_lines:
            f.write(code_lines)




class FCodeGen(CodeGen):
    """Generator for Fortran 95 code

    The .write() method inherited from CodeGen will output a code file and
    an interface file, <prefix>.f90 and <prefix>.h respectively.

    """

    code_extension = "f90"
    interface_extension = "h"

    def __init__(self, project='project'):
        CodeGen.__init__(self, project)

    def _get_symbol(self, s):
        """Returns the symbol as fcode prints it."""
        return fcode(s).strip()

    def _get_header(self):
        """Writes a common header for the generated files."""
        code_lines = []
        code_lines.append("!" + "*"*78 + '\n')
        tmp = header_comment % {"version": sympy_version,
            "project": self.project}
        for line in tmp.splitlines():
            code_lines.append("!*%s*\n" % line.center(76))
        code_lines.append("!" + "*"*78 + '\n')
        return code_lines

    def _preprocessor_statements(self, prefix):
        return []

    def _get_routine_opening(self, routine):
        """Returns the opening statements of the fortran routine."""
        code_list = []
        if len(routine.results) > 1:
            raise CodeGenError(
                "Fortran only supports a single or no return value.")
        elif len(routine.results) == 1:
            result = routine.results[0]
            code_list.append(result.get_datatype('fortran'))
            code_list.append("function")
        else:
            code_list.append("subroutine")

        args = ", ".join("%s" % self._get_symbol(arg.name)
                        for arg in routine.arguments)

        call_sig = "{0}({1})\n".format(routine.name, args)
        # Fortran 95 requires all lines be less than 132 characters, so wrap
        # this line before appending.
        call_sig = ' &\n'.join(textwrap.wrap(call_sig,
                                             width=60,
                                             break_long_words=False)) + '\n'
        code_list.append(call_sig)
        code_list = [' '.join(code_list)]
        code_list.append('implicit none\n')
        return code_list

    def _declare_arguments(self, routine):
        # argument type declarations
        code_list = []
        array_list = []
        scalar_list = []
        for arg in routine.arguments:

            if isinstance(arg, InputArgument):
                typeinfo = "%s, intent(in)" % arg.get_datatype('fortran')
            elif isinstance(arg, InOutArgument):
                typeinfo = "%s, intent(inout)" % arg.get_datatype('fortran')
            elif isinstance(arg, OutputArgument):
                typeinfo = "%s, intent(out)" % arg.get_datatype('fortran')
            else:
                raise CodeGenError("Unkown Argument type: %s" % type(arg))

            fprint = self._get_symbol

            if arg.dimensions:
                # fortran arrays start at 1
                dimstr = ", ".join(["%s:%s" % (
                    fprint(dim[0] + 1), fprint(dim[1] + 1))
                    for dim in arg.dimensions])
                typeinfo += ", dimension(%s)" % dimstr
                array_list.append("%s :: %s\n" % (typeinfo, fprint(arg.name)))
            else:
                scalar_list.append("%s :: %s\n" % (typeinfo, fprint(arg.name)))

        # scalars first, because they can be used in array declarations
        code_list.extend(scalar_list)
        code_list.extend(array_list)

        return code_list

    def _declare_globals(self, routine):
        # Global variables not explicitly declared within Fortran 90 functions.
        # Note: a future F77 mode may need to generate "common" blocks.
        return []

    def _declare_locals(self, routine):
        code_list = []
        for var in sorted(routine.local_vars, key=str):
            typeinfo = get_default_datatype(var)
            code_list.append("%s :: %s\n" % (
                typeinfo.fname, self._get_symbol(var)))
        return code_list

    def _get_routine_ending(self, routine):
        """Returns the closing statements of the fortran routine."""
        if len(routine.results) == 1:
            return ["end function\n"]
        else:
            return ["end subroutine\n"]

    def get_interface(self, routine):
        """Returns a string for the function interface.

        The routine should have a single result object, which can be None.
        If the routine has multiple result objects, a CodeGenError is
        raised.

        See: http://en.wikipedia.org/wiki/Function_prototype

        """
        prototype = [ "interface\n" ]
        prototype.extend(self._get_routine_opening(routine))
        prototype.extend(self._declare_arguments(routine))
        prototype.extend(self._get_routine_ending(routine))
        prototype.append("end interface\n")

        return "".join(prototype)

    def _call_printer(self, routine):
        declarations = []
        code_lines = []

        variables = list(routine.statements) + routine.result_variables

        for result in variables:
            expr = None
            skip = False
            if isinstance(result, Result):
                assign_to = routine.name
                expr      = result.expr
            elif isinstance(result, (OutputArgument, InOutArgument)):
                assign_to = result.result_var
                expr      = result.expr
                try:
                    if Eq(expr, assign_to):
                        skip = True
                except:
                    pass
            elif isinstance(result, Assign):
                assign_to = result.lhs
                expr      = result.rhs
            else:
                raise ValueError("Unknown variable : %s" % result)

            if not skip:
                constants   = set([])
                not_fortran = set([])
                f_expr      = ""
                if isinstance(expr, For):
                    f_expr = fcode(expr, source_format='free', human=False)
                else:
                    if isinstance(result, Assign):
                        f_expr = fcode(expr, assign_to=assign_to, source_format='free', human=False)
                    else:
                        constants, not_fortran, f_expr = fcode(expr,
                            assign_to=assign_to, source_format='free', human=False)

                for obj, v in sorted(constants, key=str):
                    t = get_default_datatype(obj)
                    declarations.append(
                        "%s, parameter :: %s = %s\n" % (t.fname, obj, v))
                for obj in sorted(not_fortran, key=str):
                    t = get_default_datatype(obj)
                    if isinstance(obj, Function):
                        name = obj.func
                    else:
                        name = obj
                    declarations.append("%s :: %s\n" % (t.fname, name))

                code_lines.append("%s\n" % f_expr)

        return declarations + code_lines

    def _indent_code(self, codelines):
        p = FCodePrinter({'source_format': 'free', 'human': False})
        return p.indent_code(codelines)

    def dump_f95(self, routines, f, prefix, header=True, empty=True):
        # check that symbols are unique with ignorecase
        for r in routines:
            lowercase = {str(x).lower() for x in r.variables}
            orig_case = {str(x) for x in r.variables}
            if len(lowercase) < len(orig_case):
                raise CodeGenError("Fortran ignores case. Got symbols: %s" %
                        (", ".join([str(var) for var in r.variables])))
        self.dump_code(routines, f, prefix, header, empty)

    dump_f95.extension = code_extension
    dump_f95.__doc__ = CodeGen.dump_code.__doc__

    def dump_h(self, routines, f, prefix, header=True, empty=True):
        """Writes the interface to a header file.

        This file contains all the function declarations.

        Parameters
        ==========

        routines : list
            A list of Routine instances.

        f : file-like
            Where to write the file.

        prefix : string
            The filename prefix.

        header : bool, optional
            When True, a header comment is included on top of each source
            file.  [default : True]

        empty : bool, optional
            When True, empty lines are included to structure the source
            files.  [default : True]

        """
        if header:
            print(''.join(self._get_header()), file=f)
        if empty:
            print(file=f)
        # declaration of the function prototypes
        for routine in routines:
            prototype = self.get_interface(routine)
            f.write(prototype)
        if empty:
            print(file=f)

    dump_h.extension = interface_extension

    # This list of dump functions is used by CodeGen.write to know which dump
    # functions it has to call.
    dump_fns = [dump_f95, dump_h]


class LuaCodeGen(CodeGen):
    """Generator for Lua code.

    The .write() method inherited from CodeGen will output a code file
    <prefix>.lua

    """

    code_extension = "lua"

    def routine(self, name, expr, argument_sequence, statements, global_vars, \
                local_vars=None, return_vars=None):
        """Specialized Routine creation for Lua."""

        if is_sequence(expr) and not isinstance(expr, (MatrixBase, MatrixExpr)):
            if not expr:
                raise ValueError("No expression given")
            expressions = Tuple(*expr)
        else:
            expressions = Tuple(expr)

        # global variables
        global_vars = set() if global_vars is None else set(global_vars)

        # symbols that should be arguments
        free_symbols = set([])
        for expr in expressions:
            if not isinstance(expr, For):
                for f in expr.free_symbols:
                    free_symbols.add(f)
            else:
                free_symbols.add(expr.iterable.stop)
                if not isinstance(expr.target, Idx):
                    free_symbols.add(expr.target)


        # local variables
        if local_vars is None:
            local_vars = set([])
        local_vars = local_vars.union({i.label for i in expressions.atoms(Idx)})

        symbols = free_symbols - local_vars - global_vars


        # Lua supports multiple return values
        return_vals = []
        if not return_vars is None:
            return_vals += list(return_vars)

        output_args = []
        stmts = []
        for (i, expr) in enumerate(expressions):
            if isinstance(expr, Equality):
                out_arg = expr.lhs
                expr = expr.rhs
                symbol = out_arg
                if isinstance(out_arg, Indexed):
                    dims = tuple([ (S.One, dim) for dim in out_arg.shape])
                    symbol = out_arg.base.label
                    output_args.append(InOutArgument(symbol, out_arg, expr, dimensions=dims))
                if not isinstance(out_arg, (Indexed, Symbol, MatrixSymbol)):
                    raise CodeGenError("Only Indexed, Symbol, or MatrixSymbol "
                                       "can define output arguments.")

                return_vals.append(Result(expr, name=symbol, result_var=out_arg))
                if not expr.has(symbol):
                    # this is a pure output: remove from the symbols list, so
                    # it doesn't become an input.
                    symbols.remove(symbol)

            elif isinstance(expr, Assign):
                out_arg = expr.lhs
                expr = expr.rhs
                if isinstance(out_arg, Indexed):
                    dims = tuple([ (S.Zero, dim - 1) for dim in out_arg.shape])
                    symbol = out_arg.base.label
                elif isinstance(out_arg, Symbol):
                    dims = []
                    symbol = out_arg
                elif isinstance(out_arg, MatrixSymbol):
                    dims = tuple([ (S.Zero, dim - 1) for dim in out_arg.shape])
                    symbol = out_arg
                else:
                    raise CodeGenError("Only Indexed, Symbol, or MatrixSymbol "
                                       "can define output arguments.")

#                if out_arg in argument_sequence:
#                    #return_vals.append(Result(expr, name=out_arg))
#                    return_vals.append(Result(expr, name=symbol, result_var=out_arg))
#                else:
#                    stmts.append(Assign(out_arg, expr))
                stmts.append(Assign(out_arg, expr))

                if (expr.has(symbol)) or (not(symbol in local_vars)):
                    output_args.append(
                        InOutArgument(symbol, out_arg, out_arg + S.Zero, dimensions=dims))

                # avoid duplicate arguments
                if not(symbol in local_vars):
                    symbols.remove(symbol)

            elif isinstance(expr, For):
                stmts.append(expr)
            else:
                # we have no name for this output
                return_vals.append(Result(expr, name='out%d' % (i+1)))

        # setup input argument list
        output_args.sort(key=lambda x: str(x.name))
        arg_list = list(output_args)
        array_symbols = {}
        for array in expressions.atoms(Indexed):
            array_symbols[array.base.label] = array
        for array in expressions.atoms(MatrixSymbol):
            array_symbols[array] = array

        for symbol in sorted(symbols, key=str):
            arg_list.append(InputArgument(symbol))

        if argument_sequence is not None:
            # if the user has supplied IndexedBase instances, we'll accept that
            new_sequence = []
            for arg in argument_sequence:
                if isinstance(arg, IndexedBase):
                    new_sequence.append(arg.label)
                else:
                    new_sequence.append(arg)
            argument_sequence = new_sequence

            missing = [x for x in arg_list if x.name not in argument_sequence]
            if missing:
                msg = "Argument list didn't specify: {0} "
                msg = msg.format(", ".join([str(m.name) for m in missing]))
                raise CodeGenArgumentListError(msg, missing)

            # create redundant arguments to produce the requested sequence
            name_arg_dict = dict([(x.name, x) for x in arg_list])
            new_args = []
            for symbol in argument_sequence:
                try:
                    new_args.append(name_arg_dict[symbol])
                except KeyError:
                    new_args.append(InputArgument(symbol))
            arg_list = new_args

        arg_list.sort(key=lambda x: str(x.name).lower())
        return Routine(name, arg_list, return_vals, stmts, local_vars, global_vars)

    def _get_header(self):
        """Writes a common header for the generated files."""
        code_lines = []
        code_lines.append("/*\n")
        tmp = header_comment % {"version": sympy_version,
                                "project": self.project}
        for line in tmp.splitlines():
            code_lines.append((" *%s" % line.center(76)).rstrip() + "\n")
        code_lines.append(" */\n")
        return code_lines

    def get_prototype(self, routine):
        """Returns a string for the function prototype of the routine.

        If the routine has multiple result objects, an CodeGenError is
        raised.

        See: http://en.wikipedia.org/wiki/Function_prototype

        """
        results = [i.get_datatype('Lua') for i in routine.results]

        # no type specification for results
        rstype = ""

        type_args = []
        for arg in routine.arguments:
            name = lua_code(arg.name)
            if arg.dimensions or isinstance(arg, ResultBase):
                type_args.append(("%s" % name, arg.get_datatype('Lua')))
            else:
                type_args.append((name, arg.get_datatype('Lua')))
        arguments = ", ".join([ "%s" % t[0] for t in type_args])
        return "function %s(%s)%s" % (routine.name, arguments, rstype)

    def _preprocessor_statements(self, prefix):
        code_lines = []
        # code_lines.append("use std::f64::consts::*;\n")
        return code_lines

    def _get_routine_opening(self, routine):
        prototype = self.get_prototype(routine)
        return ["%s \n" % prototype]

    def _get_routine_ending(self, routine):
        """Returns the closing statements of the lua routine."""
        return ["end\n"]

    def _declare_arguments(self, routine):
        # arguments are declared in prototype
        return []

    def _declare_globals(self, routine):
        # global variables are not explicitly declared within C functions
        return []

    def _declare_locals(self, routine):
        return []

    def _call_printer(self, routine):

        code_lines = []
        declarations = []
        returns = []

        # Compose a list of symbols to be dereferenced in the function
        # body. These are the arguments that were passed by a reference
        # pointer, excluding arrays.
        dereference = []
        for arg in routine.arguments:
            if isinstance(arg, ResultBase) and not arg.dimensions:
                dereference.append(arg.name)

        def _construct_results(expr):
            _results = []
            if isinstance(expr, Result):
                if not isinstance(expr.result_var, Idx):
                    _results.append(expr.result_var)
            elif isinstance(expr, Assign):
                if not isinstance(expr.lhs, Idx):
                    _results.append(expr.lhs)
            elif isinstance(expr, For):
                # look inside For statements, recursively
                for _expr in expr.body:
                    _results += _construct_results(_expr)
            else:
                raise ValueError("Unknown variable : %s" % expr)
            return _results

        results = []
        for e in routine.results + routine.statements:
            results += _construct_results(e)

        local_vars = routine.local_vars

        # TODO for the moment, For is passed as a Result. must be changed, first
        # in the routine method...
        stmts = list(routine.statements) + routine.results
        for i, result in enumerate(stmts):
            expr      = None
            assign_to = None
            if isinstance(result, Result):
                assign_to = result.result_var
                expr      = result.expr

                lua_expr = lua_code(expr, assign_to=assign_to, human=False,
                                    local_vars=local_vars)

                if assign_to in results:
                    code_lines.append("%s\n" % lua_expr);
                else:
                    code_lines.append("local %s\n" % lua_expr);

                local_vars = set([x for x in local_vars \
                                  if (not str(x) == str(assign_to))])
            elif isinstance(result, Assign):
                assign_to = result.lhs
                expr      = result.rhs

                lua_expr = lua_code(expr, assign_to=assign_to, human=False,
                                    local_vars=local_vars)

                if assign_to in results:
                    code_lines.append("%s\n" % lua_expr);
                else:
                    code_lines.append("local %s\n" % lua_expr);

                local_vars = set([x for x in local_vars \
                                  if (not str(x) == str(assign_to))])
            elif isinstance(result, For):
                lua_expr = lua_code(result, human=False,
                                    local_vars=local_vars)
                code_lines.append("%s\n" % lua_expr);
            else:
                raise ValueError("Unknown variable : %s" % result)

        def _construct_returns(expr):
            _returns = []
            if isinstance(expr, Result) and \
               (not expr.result_var in routine.local_vars) and \
               (not isinstance(expr.result_var, Idx)):
                _returns.append(str(expr.result_var))
            elif isinstance(expr, Assign) and \
               (not expr.lhs in routine.local_vars) and \
               (not isinstance(expr.lhs, Idx)):
                _returns.append(str(expr.lhs))
            elif isinstance(expr, For):
                # look inside For statements, recursively
                for _expr in expr.body:
                    _returns += _construct_returns(_expr)
            return _returns

        returns = []
        for e in routine.results + routine.statements:
            returns += _construct_returns(e)

        returns = list(set(returns))

        if len(returns) == 1:
            returns = ['return ' + str(returns[0]) ]
        elif len(returns) > 1:
            returns.sort()
            returns = ['return {' + ', '.join(returns) + '}']
        returns.append('\n')

        # update integers: lua does not have an integer type
        # we must use the floor function
        integers = []
        for arg in routine.arguments:
            if isinstance(arg, Argument) and not arg.dimensions:
                if arg.name.is_integer:
                    arg_code = str(arg.name)
                    dcl = "local " + arg_code + " = " + \
                          "math.floor(" + arg_code + ")\n"

                    declarations.append(dcl)

        return declarations + code_lines + returns

    def _indent_code(self, codelines):
        p = LuaCodePrinter()
        return p.indent_code(codelines)

    def dump_lua(self, routines, f, prefix, header=True, empty=True):
        self.dump_code(routines, f, prefix, header, empty)

    dump_lua.extension = code_extension
    dump_lua.__doc__ = CodeGen.dump_code.__doc__

    # This list of dump functions is used by CodeGen.write to know which dump
    # functions it has to call.
    dump_fns = [dump_lua]

def get_code_generator(language, project):

    CodeGenClass = {"LUA": LuaCodeGen,
                    "F95": FCodeGen}.get(language.upper())
    if CodeGenClass is None:
        raise ValueError("Language '%s' is not supported." % language)
    return CodeGenClass(project)


#
# Friendly functions
#


def codegen(name_expr, language, prefix=None, project="project",
            to_files=False, header=True, empty=True, argument_sequence=None,
            global_vars=None, local_vars=None, return_vars=None, statements=None):
    """Generate source code for expressions in a given language.

    Parameters
    ==========

    name_expr : tuple, or list of tuples
        A single (name, expression) tuple or a list of (name, expression)
        tuples.  Each tuple corresponds to a routine.  If the expression is
        an equality (an instance of class Equality) the left hand side is
        considered an output argument.  If expression is an iterable, then
        the routine will have multiple outputs.

    language : string
        A string that indicates the source code language.  This is case
        insensitive.  Currently, 'C', 'F95' and 'Octave' are supported.
        'Octave' generates code compatible with both Octave and Matlab.

    prefix : string, optional
        A prefix for the names of the files that contain the source code.
        Language-dependent suffixes will be appended.  If omitted, the name
        of the first name_expr tuple is used.

    project : string, optional
        A project name, used for making unique preprocessor instructions.
        [default: "project"]

    to_files : bool, optional
        When True, the code will be written to one or more files with the
        given prefix, otherwise strings with the names and contents of
        these files are returned. [default: False]

    header : bool, optional
        When True, a header is written on top of each source file.
        [default: True]

    empty : bool, optional
        When True, empty lines are used to structure the code.
        [default: True]

    argument_sequence : iterable, optional
        Sequence of arguments for the routine in a preferred order.  A
        CodeGenError is raised if required arguments are missing.
        Redundant arguments are used without warning.  If omitted,
        arguments will be ordered alphabetically, but with all input
        aguments first, and then output or in-out arguments.

    statements: list of Statements
        List of statements (for example Assign of local variables)

    global_vars : iterable, optional
        Sequence of global variables used by the routine.  Variables
        listed here will not show up as function arguments.

    local_vars : iterable, optional
        Sequence of local variables used by the routine.  Variables
        listed here will not show up as function arguments.

    return_vars : iterable, optional
        Sequence of returned variables for the routine.  Variables
        listed here will not show up as function arguments.

    Examples
    ========

    >>> from sympy.utilities.codegen import codegen
    >>> from sympy.abc import x, y, z
    >>> [(c_name, c_code), (h_name, c_header)] = codegen(
    ...     ("f", x+y*z), "C89", "test", header=False, empty=False)
    >>> print(c_name)
    test.c
    >>> print(c_code)
    #include "test.h"
    #include <math.h>
    double f(double x, double y, double z) {
       double f_result;
       f_result = x + y*z;
       return f_result;
    }
    <BLANKLINE>
    >>> print(h_name)
    test.h
    >>> print(c_header)
    #ifndef PROJECT__TEST__H
    #define PROJECT__TEST__H
    double f(double x, double y, double z);
    #endif
    <BLANKLINE>

    Another example using Equality objects to give named outputs.  Here the
    filename (prefix) is taken from the first (name, expr) pair.

    >>> from sympy.abc import f, g
    >>> from sympy import Eq
    >>> [(c_name, c_code), (h_name, c_header)] = codegen(
    ...      [("myfcn", x + y), ("fcn2", [Eq(f, 2*x), Eq(g, y)])],
    ...      "C99", header=False, empty=False)
    >>> print(c_name)
    myfcn.c
    >>> print(c_code)
    #include "myfcn.h"
    #include <math.h>
    double myfcn(double x, double y) {
       double myfcn_result;
       myfcn_result = x + y;
       return myfcn_result;
    }
    void fcn2(double x, double y, double *f, double *g) {
       (*f) = 2*x;
       (*g) = y;
    }
    <BLANKLINE>

    If the generated function(s) will be part of a larger project where various
    global variables have been defined, the 'global_vars' option can be used
    to remove the specified variables from the function signature

    >>> from sympy.utilities.codegen import codegen
    >>> from sympy.abc import x, y, z
    >>> [(f_name, f_code), header] = codegen(
    ...     ("f", x+y*z), "F95", header=False, empty=False,
    ...     argument_sequence=(x, y), global_vars=(z,))
    >>> print(f_code)
    REAL*8 function f(x, y)
    implicit none
    REAL*8, intent(in) :: x
    REAL*8, intent(in) :: y
    f = x + y*z
    end function
    <BLANKLINE>

    """

    # Initialize the code generator.
    code_gen = get_code_generator(language, project)

    if isinstance(name_expr[0], string_types):
        # single tuple is given, turn it into a singleton list with a tuple.
        name_expr = [name_expr]

    if prefix is None:
        prefix = name_expr[0][0]

    # Construct Routines appropriate for this code_gen from (name, expr) pairs.
    routines = []
    for name, expr in name_expr:
        routines.append(code_gen.routine(name, expr, argument_sequence, statements, global_vars, \
                                         local_vars=local_vars, \
                                         return_vars=return_vars))

    # Write the code.
    return code_gen.write(routines, prefix, to_files, header, empty)


def make_routine(name, expr, argument_sequence=None,
                 global_vars=None, language="F95", local_vars=None):
    """A factory that makes an appropriate Routine from an expression.

    Parameters
    ==========

    name : string
        The name of this routine in the generated code.

    expr : expression or list/tuple of expressions
        A SymPy expression that the Routine instance will represent.  If
        given a list or tuple of expressions, the routine will be
        considered to have multiple return values and/or output arguments.

    argument_sequence : list or tuple, optional
        List arguments for the routine in a preferred order.  If omitted,
        the results are language dependent, for example, alphabetical order
        or in the same order as the given expressions.

    global_vars : iterable, optional
        Sequence of global variables used by the routine.  Variables
        listed here will not show up as function arguments.

    language : string, optional
        Specify a target language.  The Routine itself should be
        language-agnostic but the precise way one is created, error
        checking, etc depend on the language.  [default: "F95"].

    local_vars : iterable, optional
        Sequence of local variables used by the routine.  Variables
        listed here will not show up as function arguments.

    A decision about whether to use output arguments or return values is made
    depending on both the language and the particular mathematical expressions.
    For an expression of type Equality, the left hand side is typically made
    into an OutputArgument (or perhaps an InOutArgument if appropriate).
    Otherwise, typically, the calculated expression is made a return values of
    the routine.

    Examples
    ========

    >>> from sympy.utilities.codegen import make_routine
    >>> from sympy.abc import x, y, f, g
    >>> from sympy import Eq
    >>> r = make_routine('test', [Eq(f, 2*x), Eq(g, x + y)])
    >>> [arg.result_var for arg in r.results]
    []
    >>> [arg.name for arg in r.arguments]
    [x, y, f, g]
    >>> [arg.name for arg in r.result_variables]
    [f, g]
    >>> r.local_vars
    set()

    Another more complicated example with a mixture of specified and
    automatically-assigned names.  Also has Matrix output.

    >>> from sympy import Matrix
    >>> r = make_routine('fcn', [x*y, Eq(f, 1), Eq(g, x + g), Matrix([[x, 2]])])
    >>> [arg.result_var for arg in r.results]  # doctest: +SKIP
    [result_5397460570204848505]
    >>> [arg.expr for arg in r.results]
    [x*y]
    >>> [arg.name for arg in r.arguments]  # doctest: +SKIP
    [x, y, f, g, out_8598435338387848786]

    We can examine the various arguments more closely:

    >>> from sympy.utilities.codegen import (InputArgument, OutputArgument,
    ...                                      InOutArgument)
    >>> [a.name for a in r.arguments if isinstance(a, InputArgument)]
    [x, y]

    >>> [a.name for a in r.arguments if isinstance(a, OutputArgument)]  # doctest: +SKIP
    [f, out_8598435338387848786]
    >>> [a.expr for a in r.arguments if isinstance(a, OutputArgument)]
    [1, Matrix([[x, 2]])]

    >>> [a.name for a in r.arguments if isinstance(a, InOutArgument)]
    [g]
    >>> [a.expr for a in r.arguments if isinstance(a, InOutArgument)]
    [g + x]

    """

    # initialize a new code generator
    code_gen = get_code_generator(language, "nothingElseMatters")

    return code_gen.routine(name, expr, argument_sequence, global_vars, \
                            local_vars=local_vars)


class CodeGenError(Exception):
    pass


class CodeGenArgumentListError(Exception):
    @property
    def missing_args(self):
        return self.args[1]


header_comment = """Code generated with sympy %(version)s

See http://www.sympy.org/ for more information.

This file is part of '%(project)s'
"""
