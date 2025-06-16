"""Classes and methods that handle supported Openmp objects"""
import re
from ast import AST

from pyccel.ast.basic import PyccelAstNode
from pyccel.errors.errors import Errors

errors = Errors()

__all__ = (
    "OmpAst",
    "OmpClause",
    "OmpConstantPositiveInteger",
    "OmpConstruct",
    "OmpDirective",
    "OmpEndDirective",
    "OmpExpr",
    "OmpExpressionList",
    "OmpIntegerExpr",
    "OmpList",
    "OmpNode",
    "OmpScalarExpr",
    "OmpTxClause",
    "OmpTxConstantPositiveInteger",
    "OmpTxDirective",
    "OmpTxEndDirective",
    "OmpTxExpr",
    "OmpTxExpressionList",
    "OmpTxIntegerExpr",
    "OmpTxList",
    "OmpTxNode",
    "OmpTxScalarExpr",
)


class OmpAst(AST):
    """
    New AST node representing an OPENMP syntax.

    This class extends the AST class to represent OpenMP syntax in the abstract syntax tree.

    Parameters
    ----------
    lineno : int
        Line number in the source code where the OpenMP syntax occurs.
    col_offset : int
        Column offset in the source code where the OpenMP syntax occurs.

    See Also
    --------
    OmpNode : Parent class for all OpenMP classes.

    Examples
    --------
    >>> node = OmpAst(10, 5)
    >>> node.lineno
    10
    >>> node.col_offset
    5
    """

    def __init__(self, lineno, col_offset):
        super().__init__()
        self.lineno = lineno
        self.col_offset = col_offset


class OmpNode(PyccelAstNode):
    """
    Parent class for all openmp classes, including classes used in the grammar.

    This is the base class for all OpenMP-related classes in the codebase.
    It provides common functionality for handling OpenMP directives and constructs.

    Parameters
    ----------
    raw : str
        The raw syntax of the comment's section that represents the object, 
        as provided in the source code.
    position : tuple
        The start and end positions of the OpenMP syntax in the source code, 
        used to print errors.
    **kwargs : dict
        Additional keyword arguments to be passed to the parent class.

    See Also
    --------
    OmpConstruct : Class representing an OpenMP construct.
    OmpDirective : Class representing an OpenMP directive.
    OmpClause : Class representing an OpenMP clause.

    Examples
    --------
    >>> s = "omp parallel"
    >>> node = OmpNode(raw=s, position=(5, len(s)))
    """

    _attribute_nodes = ()

    def __init__(self, raw, position, **kwargs):
        super().__init__()
        self._raw = raw
        self._position = position

    @property
    def position(self):
        """
        Returns the position_start and position_end of an omp object's syntax inside the pragma.
        """
        return self._position

    @property
    def raw(self):
        """
        Finds root model of the omp object and returns the object's syntax as written in the code.
        """
        return self._raw

    def get_fixed_state(self):
        """
        Get the attributes that do not change throughout the life of the object.

        This method returns a dictionary containing the immutable attributes
        of an OpenMP object.

        Returns
        -------
        dict
            Dictionary containing position and raw attributes.

        See Also
        --------
        position : Property that returns the position of the syntax.
        raw : Property that returns the raw syntax.
        """
        return {
            'position': self.position,
            # 'line': self.line,
            'raw': self.raw,
        }


class OmpConstruct(OmpNode):
    """
    Represents an OpenMP Construct.

    An OpenMP construct is a block of code, delimited by an OpenMP Directive at the
    start and potentially an end directive. It encapsulates a section of code that
    should be processed according to OpenMP rules.

    Parameters
    ----------
    start : OmpDirective
        The directive that marks the start of the construct.
    body : CodeBlock
        The block of code (code statements) or syntax forming the body within the construct.
    end : OmpEndDirective, optional
        The directive that marks the end of the construct, if applicable.

    See Also
    --------
    OmpDirective : Class representing the directive that starts a construct.
    OmpEndDirective : Class representing the directive that ends a construct.

    Examples
    --------
    >>> from pyccel.ast.core import CodeBlock
    >>> start_directive = OmpDirective(name="parallel", clauses=(), is_construct=True,
    ...                             raw="omp parallel", position=(4, 17))
    >>> end_directive = OmpEndDirective(name="parallel", clauses=(), is_construct=True,
    ...                             raw="omp end parallel", position=(4, 21))
    >>> body = CodeBlock(body = [])
    >>> construct = OmpConstruct(start=start_directive, body=body, end=end_directive)
    """
    _attribute_nodes = ("_start", "_end", "_body")

    def __init__(self, start, body, end=None):
        self._start = start
        self._end = end
        self._body = body
        super().__init__(raw=start.raw, position=start.position)

    @property
    def start(self):
        """Returns the directive that marks the start of the construct"""
        return self._start

    @property
    def body(self):
        """
        Returns a codeblock body of the construct.
        """
        return self._body

    @property
    def end(self):
        """
        Returns the end directive that marks the end of the construct.
        """
        return self._end


class OmpDirective(OmpNode):
    """
    Represents an OpenMP Directive.

    An OpenMP directive is a specific instruction to the compiler that indicates
    how to handle a section of code according to OpenMP rules.

    Parameters
    ----------
    name : str
        The name of the directive.
    clauses : tuple of OmpClause
        Clauses passed to the directive.
    is_construct : bool
        True if the directive is syntactically incorrect without a corresponding end directive.
    raw : str
        The raw syntax of the directive as provided in the source code.
    position : tuple
        The start and end positions of the directive in the source code.
    **kwargs : dict
        Additional keyword arguments to be passed to the parent class.

    See Also
    --------
    OmpConstruct : Class representing an OpenMP construct.
    OmpClause : Class representing clauses that can be part of a directive.
    OmpEndDirective : Class representing the end of a directive.

    Examples
    --------
    >>> directive = OmpDirective(name="parallel", clauses=(), is_construct=True, 
    ...                         raw="omp parallel", position=(8, 21))
    """
    _attribute_nodes = ("_clauses",)

    def __init__(self, name, clauses, is_construct, raw,
                 position, **kwargs):
        self._name = name
        self._is_construct = is_construct
        self._clauses = clauses
        super().__init__(raw, position, **kwargs)

    @property
    def name(self):
        """
        Returns the name of the directive.
        """
        return self._name

    @property
    def clauses(self):
        """
        Returns the clauses of the directive.
        """
        return self._clauses

    @property
    def is_construct(self):
        """
        Returns True if the directive is a construct.
        """
        return self._is_construct

    def get_fixed_state(self):
        """
        Get the attributes that do not change throughout the life of the directive.

        This method returns a dictionary containing the immutable attributes
        of an OpenMP directive.

        Returns
        -------
        dict
            Dictionary containing position, raw, name, and is_construct attributes.

        See Also
        --------
        name : Property that returns the name of the directive.
        is_construct : Property that indicates if the directive is a construct.

        Examples
        --------
        >>> directive = OmpDirective(name="parallel", clauses=(), is_construct=True, 
        ...                         raw="omp parallel", position=(8, 21))
        >>> state = directive.get_fixed_state()
        >>> state['name']
        'parallel'
        """
        d_fixed = super().get_fixed_state()
        return {
            **d_fixed,
            'name': self.name,
            'is_construct': self.is_construct,
        }


class OmpEndDirective(OmpDirective):
    """
    Represents an OpenMP End Directive.

    An OpenMP End Directive marks the end of an OpenMP construct. It is paired with
    a corresponding start directive to delimit a block of code that should be processed
    according to OpenMP rules.

    Parameters
    ----------
    name : str
        The name of the directive.
    clauses : tuple of OmpClause
        Clauses passed to the directive.
    is_construct : bool
        True if the directive is syntactically incorrect without a corresponding end directive.
    raw : str
        The raw syntax of the directive as provided in the source code.
    position : tuple
        The start and end positions of the directive in the source code.
    **kwargs : dict
        Additional keyword arguments to be passed to the parent class.

    See Also
    --------
    OmpDirective : Parent class for directives.
    OmpConstruct : Class representing an OpenMP construct.

    Examples
    --------
    >>> end_directive = OmpEndDirective(name="parallel", clauses=(), is_construct=True,
    ...                               raw="omp end parallel", position=(8, 25))
    """


class OmpClause(OmpNode):
    """
    Represents an OpenMP Clause.

    An OpenMP clause modifies the behavior of an OpenMP directive by providing
    additional information to the compiler.

    Parameters
    ----------
    name : str
        The name of the clause.
    omp_exprs : tuple
        OpenMP expressions passed to the clause.
    raw : str
        The raw syntax of the clause as provided in the source code.
    position : tuple
        The start and end positions of the clause in the source code.
    **kwargs : dict
        Additional keyword arguments to be passed to the parent class.

    See Also
    --------
    OmpDirective : Class that can contain clauses.
    OmpExpr : Class representing expressions that can be part of a clause.

    Examples
    --------
    >>> clause = OmpClause(name="private", omp_exprs=(),
    ...                   raw="private(x, y)", position=(4, 17))
    """

    _attribute_nodes = ("_omp_exprs",)

    def __init__(self, name, omp_exprs, raw, position, **kwargs):
        self._omp_exprs = omp_exprs
        self._name = name
        super().__init__(raw, position, **kwargs)

    @property
    def omp_exprs(self):
        """
        Returns the OpenMP expressions of the clause.
        """
        return self._omp_exprs

    @property
    def name(self):
        """
        Returns the name of the clause.
        """
        return self._name

    def get_fixed_state(self):
        """
        Get the attributes that do not change throughout the life of the clause.

        This method returns a dictionary containing the immutable attributes
        of an OpenMP clause.

        Returns
        -------
        dict
            Dictionary containing name, position, and raw attributes.

        See Also
        --------
        name : Property that returns the name of the clause.

        Examples
        --------
        >>> clause = OmpClause(name="private", omp_exprs=(), 
        ...                   raw="private(x, y)", position=(4, 17))
        >>> state = clause.get_fixed_state()
        >>> state['name']
        'private'
        """
        d_fixed = super().get_fixed_state()
        return {
            'name': self.name,
            **d_fixed,
        }


class OmpExpr(OmpNode):
    """
    Parent class of OpenMP object that represents a Python expression.

    This class serves as a base for all OpenMP expressions that can be translated
    to Python expressions.

    Parameters
    ----------
    value : any
        Pyccel object that represents the value of the object.
    raw : str
        The raw syntax of the expression as provided in the source code.
    position : tuple
        The start and end positions of the expression in the source code.
    **kwargs : dict
        Additional keyword arguments to be passed to the parent class.

    See Also
    --------
    OmpScalarExpr : Class representing a scalar expression.
    OmpList : Class representing a list expression.
    OmpIntegerExpr : Class representing an integer expression.

    Examples
    --------
    >>> from pyccel.ast.datatypes import PythonNativeInt, PythonNativeFloat
    >>> from pyccel.ast.core import Variable
    >>> n = Variable(PythonNativeInt(), 'n')
    >>> expr = OmpExpr(value=n, raw="n", position=(10, 11))
    """

    _attribute_nodes = ()

    def __init__(self, value, raw, position, **kwargs):
        self._value = value
        super().__init__(raw, position, **kwargs)

    @property
    def value(self):
        """
        Returns the value of the expression.
        """
        return self._value


class OmpScalarExpr(OmpExpr):
    """
    Represents a scalar expression in OpenMP.

    This class is used for scalar values in OpenMP expressions, such as
    variables or constants.

    Parameters
    ----------
    value : any
        Pyccel objects that represent the value of the expression.
    raw : str
        The raw syntax of the expression as provided in the source code.
    position : tuple
        The start and end positions of the expression in the source code.
    **kwargs : dict
        Additional keyword arguments to be passed to the parent class.

    See Also
    --------
    OmpExpr : Parent class for all OpenMP expressions.
    OmpIntegerExpr : Class for integer expressions.

    Examples
    --------
    >>> from pyccel.ast.datatypes import PythonNativeInt, PythonNativeFloat
    >>> from pyccel.ast.core import Variable
    >>> n = Variable(PythonNativeInt(), 'n')
    >>> expr = OmpScalarExpr(value=n, raw="n", position=(10, 11))
    """


class OmpConstantPositiveInteger(OmpExpr):
    """
    Represents a constant positive integer in OpenMP.

    This class is used for positive integer constants in OpenMP expressions,
    such as thread counts or chunk sizes.

    Parameters
    ----------
    value : int or str
        The positive integer value or expression.
    raw : str
        The raw syntax of the expression as provided in the source code.
    position : tuple
        The start and end positions of the expression in the source code.
    **kwargs : dict
        Additional keyword arguments to be passed to the parent class.

    See Also
    --------
    OmpExpr : Parent class for all OpenMP expressions.
    OmpIntegerExpr : Class for general integer expressions.

    Examples
    --------
    >>> from pyccel.ast.datatypes import PythonNativeInt
    >>> from pyccel.ast.variable import Constant
    >>> import math
    >>> c = Constant(PythonNativeInt(), value=3)
    >>> const = OmpConstantPositiveInteger(value=c, raw="3", position=(10, 11))
    >>> const.value
    4
    """


class OmpIntegerExpr(OmpExpr):
    """
    Represents an integer expression in OpenMP.

    This class is used for integer expressions in OpenMP, which can be
    variables or constants.

    Parameters
    ----------
    value : any
        The Pyccel Object that represents the value of expression.
    raw : str
        The raw syntax of the expression as provided in the source code.
    position : tuple
        The start and end positions of the expression in the source code.
    **kwargs : dict
        Additional keyword arguments to be passed to the parent class.

    See Also
    --------
    OmpExpr : Parent class for all OpenMP expressions.
    OmpConstantPositiveInteger : Class for positive integer constants.

    Examples
    --------
    >>> from pyccel.ast.datatypes import PythonNativeInt, PythonNativeFloat
    >>> from pyccel.ast.core import Variable
    >>> n = Variable(PythonNativeInt(), 'n')
    >>> int_expr = OmpIntegerExpr(value=n, raw="n", position=(10, 11))
    """


class OmpList(OmpExpr):
    """
    Represents a list in OpenMP.

    This class is used for list expressions in OpenMP, such as variable lists
    in private or shared clauses.

    Parameters
    ----------
    value : tuple
        The tuple of Pyccel objects that represents the value of the expression.
    raw : str
        The raw syntax of the list as provided in the source code.
    position : tuple
        The start and end positions of the list in the source code.
    **kwargs : dict
        Additional keyword arguments to be passed to the parent class.

    See Also
    --------
    OmpExpr : Parent class for all OpenMP expressions.
    OmpExpressionList : Class for lists of expressions.

    Examples
    --------
    >>> from pyccel.ast.datatypes import PythonNativeInt
    >>> from pyccel.ast.core import Variable
    >>> n = Variable(PythonNativeInt(), 'n')
    >>> m = Variable(PythonNativeInt(), 'm')
    >>> omp_list = OmpList(value=tuple(n,m), raw="(n)", position=(10, 13))
    """

    @property
    def value(self):
        """
        Returns the expression, or the tweaked raw that should represent a python expression
        """
        if self._value:
            return self._value
        else:
            return f"({self.raw},)"


class OmpExpressionList(OmpList):
    """
    Represents a list of expressions in OpenMP.

    This class is used for lists of variables and addition/substruction expressions in OpenMP.

    Parameters
    ----------
    value : tuple
        The tuple of expressions.
    raw : str
        The raw syntax of the expression list as provided in the source code.
    position : tuple
        The start and end positions of the expression list in the source code.
    **kwargs : dict
        Additional keyword arguments to be passed to the parent class.

    See Also
    --------
    OmpList : Parent class for all OpenMP lists.
    OmpExpr : Base class for all OpenMP expressions.

    Examples
    --------
    >>> from pyccel.ast.datatypes import PythonNativeInt, PythonNativeFloat
    >>> from pyccel.ast.core import Variable
    >>> n = Variable(PythonNativeInt(), 'n')
    >>> m = Variable(PythonNativeFloat(), 'm')
    >>>from pyccel.ast.operators import PyccelAdd, PyccelMinus
    >>> ex1 = PyccelAdd(n, m)
    >>> ex2 = PyccelMinus(n, m)
    >>> expr_list = OmpExpressionList(value=(n, m, ex1, ex2), raw="(n, m, n + m, n - m)", position=(10, 30))
    """


class OmpTxNode(OmpNode):
    """
    Parent class for all TextX OpenMP classes.

    This class handles common grammar logic for TextX-based OpenMP nodes.
    TextX is used to parse OpenMP pragmas and create the corresponding AST nodes.

    Parameters
    ----------
    tx_obj : object
        A TextX object representing the parsed OpenMP syntax.
    comment : str
        The Python comment that represents the OpenMP syntax.
    omp_version : float
        The version of OpenMP chosen by the user.
    lineno : int, optional
        The line number in the source code where the OpenMP syntax occurs.
    column : int, optional
        The column number in the source code where the OpenMP syntax occurs.
    version : float or list, optional
        The OpenMP version required for the annotated comment.
    deprecated : float, optional
        The OpenMP version in which the syntax was deprecated. Defaults to infinity.
    **kwargs : dict
        Additional keyword arguments to be passed to the parent class.

    See Also
    --------
    OmpNode : Parent class for all OpenMP nodes.
    OmpTxDirective : TextX-based OpenMP directive.
    OmpTxClause : TextX-based OpenMP clause.
    """

    _attribute_nodes = ()

    def __init__(self, tx_obj, comment, omp_version, lineno=None, column=None, version=None, deprecated=None,
                 **kwargs):
        self._parent = tx_obj.parent
        self._omp_version = omp_version
        position = (tx_obj._tx_position, tx_obj._tx_position_end)
        raw = comment[position[0]:position[1]]

        self._version = version or 0.0
        self._deprecated = deprecated or float("inf")
        super().__init__(raw=raw, position=position, **kwargs)
        if lineno is not None and column is not None:
            self.set_current_ast(OmpAst(lineno, self.position[0] + column))
        if self.version > self.omp_version:
            errors.report(
                f"Syntax not supported in OpenMp version {self.omp_version}",
                symbol=self,
                severity="warning",
            )
        if self.deprecated <= self.omp_version:
            errors.report(
                f"Syntax deprecated in OpenMp version {self.omp_version}",
                symbol=self,
                severity="warning",
            )

    @property
    def version(self):
        """
        Returns the version of the OpenMP object's syntax used.
        """
        return self._version

    @property
    def deprecated(self):
        """
        Returns the deprecated version of OpenMP syntax used.
        """
        return self._deprecated

    @property
    def omp_version(self):
        """
        Returns the OpenMP version used by the user.
        """
        return self._omp_version

    @property
    def parent(self):
        """
        Returns the parent of the OpenMP object.
        """
        return self._parent


class OmpTxDirective(OmpTxNode, OmpDirective):
    """
    Represents an OpenMP TextX Directive.

    This class handles common grammar logic for TextX-based OpenMP directives.
    It processes the TextX directive object and extracts relevant information
    to create an OpenMP directive node.

    Parameters
    ----------
    tx_directive : object
        A TextX directive object representing the parsed OpenMP directive.
    comment : str
        The Python comment that represents the OpenMP syntax.
    omp_version : float
        The version of OpenMP chosen by the user.
    lineno : int, optional
        The line number in the source code where the OpenMP syntax occurs.
    column : int, optional
        The column number in the source code where the OpenMP syntax occurs.
    **kwargs : dict
        Additional keyword arguments to be passed to the parent classes.

    See Also
    --------
    OmpTxNode : Parent class for all TextX OpenMP nodes.
    OmpDirective : Parent class for all OpenMP directives.
    OmpTxEndDirective : Class representing a TextX-based end directive.
    """

    _attribute_nodes = ("_clauses",)

    def __init__(self, tx_directive, comment, omp_version, lineno=None, column=None, **kwargs):
        d_attrs = {attr_name: getattr(tx_directive, attr_name) for attr_name in tx_directive._tx_attrs}
        is_construct = d_attrs.get('is_construct')
        if is_construct is None:
            is_construct = False
        # Imposed by the grammar: clean up the tx directive object's clauses.
        clauses = d_attrs.get('_tx_clauses', [])
        # Get a list containing all the versions of the children omp objects,
        # necessary to calculate the directives versions.
        version = d_attrs.get('VERSION')
        version = max(filter(None, [*[c.VERSION for c in clauses if hasattr(c, 'VERSION')], version]), default=None)
        name = d_attrs.get('name')
        clauses = [c for c in clauses if c]
        clauses = [c.clause if hasattr(c, 'clause') else c for c in clauses]
        clauses = [OmpTxClause(c, comment, omp_version, lineno=lineno, column=column) for c in clauses]

        super().__init__(tx_obj=tx_directive, comment=comment, omp_version=omp_version, lineno=lineno, column=column, name=name, clauses=clauses,
                         is_construct=is_construct, version=version,
                         deprecated=d_attrs.get('DEPRECATED'), **kwargs)
        self._raw = re.sub(r"#\s*\$\s*omp\s*(end)?", "", comment)

        # Invalid clauses are syntactically correct omp clauses, captured within the current directive, but are invalid
        # clauses in the context of the current directive
        invalid_clauses = d_attrs.get('_invalid_clauses', [])
        if len(invalid_clauses):
            errors.report(
                f"invalid clause `{invalid_clauses[0].name}` for `{name}` directive",
                symbol=self,
                column=self._position[0],
                severity="error",
            )


class OmpTxEndDirective(OmpTxDirective):
    """
    Represents an OpenMP TextX End Directive.

    This class represents the end directive of an OpenMP construct, parsed using TextX.
    It marks the end of a code block that should be processed according to OpenMP rules.

    Parameters
    ----------
    tx_directive : object
        A TextX directive object representing the parsed OpenMP end directive.
    comment : str
        The Python comment that represents the OpenMP syntax.
    omp_version : float
        The version of OpenMP chosen by the user.
    lineno : int, optional
        The line number in the source code where the OpenMP syntax occurs.
    column : int, optional
        The column number in the source code where the OpenMP syntax occurs.
    **kwargs : dict
        Additional keyword arguments to be passed to the parent classes.

    See Also
    --------
    OmpTxDirective : Parent class for TextX-based OpenMP directives.
    OmpEndDirective : Parent class for OpenMP end directives.
    OmpConstruct : Class that uses end directives to delimit blocks of code.
    """


class OmpTxClause(OmpTxNode, OmpClause):
    """
    Represents an OpenMP TextX Clause.

    This class handles TextX-based OpenMP clauses. It processes the TextX clause
    object and extracts relevant information to create an OpenMP clause node.

    Parameters
    ----------
    tx_clause : object
        A TextX clause object representing the parsed OpenMP clause.
    comment : str
        The Python comment that represents the OpenMP syntax.
    omp_version : float
        The version of OpenMP chosen by the user.
    lineno : int, optional
        The line number in the source code where the OpenMP syntax occurs.
    column : int, optional
        The column number in the source code where the OpenMP syntax occurs.
    **kwargs : dict
        Additional keyword arguments to be passed to the parent classes.

    See Also
    --------
    OmpTxNode : Parent class for all TextX OpenMP nodes.
    OmpClause : Parent class for all OpenMP clauses.
    OmpTxDirective : Class that can contain clauses.
    """

    _attribute_nodes = ("_omp_exprs",)

    def __init__(self, tx_clause, comment, omp_version, lineno=None, column=None, **kwargs):
        d_attrs = {attr_name: getattr(tx_clause, attr_name) for attr_name in tx_clause._tx_attrs}
        omp_exprs = getattr(tx_clause, 'omp_exprs', tuple())
        if not isinstance(omp_exprs, tuple):
            omp_exprs = (omp_exprs,) if omp_exprs else tuple()
        omp_exprs = tuple(
            globals().get(ex.__class__.__name__)(ex, comment, omp_version, lineno=lineno, column=column) for ex in omp_exprs)
        allowed_parents = d_attrs.get('allowed_parents', tuple())
        allowed_parents = [set(attr.allowed_parents) for attr in [*d_attrs.values(), *omp_exprs] if
                           hasattr(attr, 'allowed_parents')] + ([set(allowed_parents)] if len(allowed_parents) else [])
        allowed_parents = set.intersection(*allowed_parents) if len(allowed_parents) > 0 else None
        super().__init__(tx_obj=tx_clause, comment=comment, omp_version=omp_version, lineno=lineno, column=column, name=d_attrs.get('name'),
                         omp_exprs=omp_exprs, version=d_attrs.get('VERSION'),
                         deprecated=d_attrs.get('DEPRECATED'), **kwargs)
        if hasattr(self.parent, 'clause'):
            self._parent = self.parent.parent

        # check if the parent directive accepts the current clause.
        if allowed_parents is not None:
            if self.parent.name not in allowed_parents:
                errors.report(
                    f"invalid syntax `{self.name}` clause for `{self.parent.name}` directive",
                    symbol=self,
                    severity="error",
                )


class OmpTxExpr(OmpTxNode, OmpExpr):
    """
    Parent class of TextX OpenMP object that represents a Python expression.

    This class serves as a base for all TextX-based OpenMP expressions that can be
    translated to Python expressions. It processes the TextX expression object and
    extracts relevant information.

    Parameters
    ----------
    tx_obj : object
        A TextX object representing the parsed OpenMP expression.
    comment : str
        The Python comment that represents the OpenMP syntax.
    omp_version : float
        The version of OpenMP chosen by the user.
    lineno : int, optional
        The line number in the source code where the OpenMP syntax occurs.
    column : int, optional
        The column number in the source code where the OpenMP syntax occurs.
    **kwargs : dict
        Additional keyword arguments to be passed to the parent classes.

    See Also
    --------
    OmpTxNode : Parent class for all TextX OpenMP nodes.
    OmpExpr : Parent class for all OpenMP expressions.
    OmpTxScalarExpr : Class for TextX-based scalar expressions.
    OmpTxList : Class for TextX-based list expressions.
    """

    _attribute_nodes = ()

    def __init__(self, tx_obj, comment, omp_version, lineno=None, column=None, **kwargs):
        d_attrs = {attr_name: getattr(tx_obj, attr_name) for attr_name in tx_obj._tx_attrs}
        super().__init__(tx_obj=tx_obj, comment=comment, omp_version=omp_version, lineno=lineno, column=column, value=d_attrs.get('value'), version=d_attrs.get('VERSION'),
                         deprecated=d_attrs.get('DEPRECATED'), **kwargs)


class OmpTxScalarExpr(OmpTxExpr, OmpScalarExpr):
    """
    Represents a scalar expression in OpenMP using TextX.

    This class is used for scalar values in TextX-based OpenMP expressions,
    such as variables or constants.

    Parameters
    ----------
    tx_expr : object
        A TextX expression object representing the parsed OpenMP scalar expression.
    comment : str
        The Python comment that represents the OpenMP syntax.
    omp_version : float
        The version of OpenMP chosen by the user.
    **kwargs : dict
        Additional keyword arguments to be passed to the parent classes.

    See Also
    --------
    OmpTxExpr : Parent class for all TextX OpenMP expressions.
    OmpScalarExpr : Parent class for scalar expressions.
    """

    def __init__(self, tx_expr, comment, omp_version, **kwargs):
        super().__init__(tx_obj=tx_expr, comment=comment, omp_version=omp_version, **kwargs)


class OmpTxConstantPositiveInteger(OmpTxExpr, OmpConstantPositiveInteger):
    """
    Represents a constant positive integer in OpenMP using TextX.

    This class is used for positive integer constants in TextX-based OpenMP expressions,
    such as thread counts or chunk sizes.

    Parameters
    ----------
    tx_expr : object
        A TextX expression object representing the parsed OpenMP positive integer.
    comment : str
        The Python comment that represents the OpenMP syntax.
    omp_version : float
        The version of OpenMP chosen by the user.
    **kwargs : dict
        Additional keyword arguments to be passed to the parent classes.

    See Also
    --------
    OmpTxExpr : Parent class for all TextX OpenMP expressions.
    OmpConstantPositiveInteger : Parent class for positive integer constants.
    """

    def __init__(self, tx_expr, comment, omp_version, **kwargs):
        super().__init__(tx_obj=tx_expr, comment=comment, omp_version=omp_version, **kwargs)


class OmpTxIntegerExpr(OmpTxExpr, OmpIntegerExpr):
    """
    Represents an integer expression in OpenMP using TextX.

    This class is used for integer expressions in TextX-based OpenMP,
    which can be variables or constants.

    Parameters
    ----------
    tx_expr : object
        A TextX expression object representing the parsed OpenMP integer expression.
    comment : str
        The Python comment that represents the OpenMP syntax.
    omp_version : float
        The version of OpenMP chosen by the user.
    **kwargs : dict
        Additional keyword arguments to be passed to the parent classes.

    See Also
    --------
    OmpTxExpr : Parent class for all TextX OpenMP expressions.
    OmpIntegerExpr : Parent class for integer expressions.
    """

    def __init__(self, tx_expr, comment, omp_version, **kwargs):
        super().__init__(tx_obj=tx_expr, comment=comment, omp_version=omp_version, **kwargs)


class OmpTxList(OmpTxExpr, OmpList):
    """
    Represents a list in OpenMP using TextX.

    This class is used for list expressions in TextX-based OpenMP,
    such as variable lists in private or shared clauses.

    Parameters
    ----------
    tx_expr : object
        A TextX expression object representing the parsed OpenMP list.
    comment : str
        The Python comment that represents the OpenMP syntax.
    omp_version : float
        The version of OpenMP chosen by the user.
    **kwargs : dict
        Additional keyword arguments to be passed to the parent classes.

    See Also
    --------
    OmpTxExpr : Parent class for all TextX OpenMP expressions.
    OmpList : Parent class for list expressions.
    """

    def __init__(self, tx_expr, comment, omp_version, **kwargs):
        super().__init__(tx_obj=tx_expr, comment=comment, omp_version=omp_version, **kwargs)


class OmpTxExpressionList(OmpTxExpr, OmpExpressionList):
    """
    Represents a list of expressions in OpenMP using TextX.

    This class is used for lists of expressions in TextX-based OpenMP,
    such as lists of variables or expressions in reduction clauses.

    Parameters
    ----------
    tx_expr : object
        A TextX expression object representing the parsed OpenMP expression list.
    comment : str
        The Python comment that represents the OpenMP syntax.
    omp_version : float
        The version of OpenMP chosen by the user.
    **kwargs : dict
        Additional keyword arguments to be passed to the parent classes.

    See Also
    --------
    OmpTxExpr : Parent class for all TextX OpenMP expressions.
    OmpExpressionList : Parent class for expression lists.
    """

    def __init__(self, tx_expr, comment, omp_version, **kwargs):
        super().__init__(tx_obj=tx_expr, comment=comment, omp_version=omp_version, **kwargs)
