#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#

from pyccel.utilities.stage import PyccelStage

from .basic import TypedAstNode
from .datatypes import FixedSizeNumericType

pyccel_stage = PyccelStage()

__all__ = (
    'FunctionalFor',
    'FunctionalMax',
    'FunctionalMin',
    'FunctionalSum',
    'GeneratorComprehension',
    'MaxLimit',
    'MinLimit',
)

#==============================================================================
class FunctionalFor(TypedAstNode):

    """
    Represents a generator expression.

    Represents any generator expression e.g:
    a = [i for i in range(10)]

    Parameters
    ----------
    loops : CodeBlock/For
        The loops contained in the expression.
    expr : PyccelAstNode
        The expression at the origin of the expression
        E.g. 'i' for '[i for i in range(10)]'.
    lhs : Variable
        The variable to which the result is assigned.
    indices : list of Variable
        All iterator targets for the for loops.
    index : Variable
        Index of result in rhs
        E.g.:
        >>> a = [i in range(1,10,2)]

        is translated to:
        >>> Dummy_0 = 0
        >>> for i in range(1,10,2):
        >>>     a[Dummy_0]=i
        >>>     Dummy_0 += 1

        Index is `Dummy_0`.
    target_type : PyccelSymbol, optional
        The type of the result of the functional for. This is useful at
        the syntactic stage to pass along the final type of the lhs (list/set/array/etc).
    operations : dict
        A dictionary mapping each type of comprehension (e.g. list, array, etc.)
        to the operation used for populating it.
    conditions : list[If|None]
        A list of filter conditions corresponding to each for-loop in the comprehension.
        Each element of this list is either an `If` instance that describes the filtering
        condition for that loop, or `None` if no condition is applied in that loop.
    """
    __slots__ = ('_loops','_expr', '_lhs', '_indices','_index', '_operations',
            '_shape','_class_type', '_target_type', '_conditions')
    _attribute_nodes = ('_loops','_expr', '_lhs', '_indices','_index')

    def __init__(
        self,
        loops,
        expr=None,
        lhs=None,
        indices=None,
        index=None,
        *,
        target_type=None,
        operations=None,
        conditions
        ):

        self._loops   = loops
        self._expr    = expr
        self._lhs     = lhs
        self._indices = indices
        self._index   = index
        self._operations = operations
        self._target_type = target_type
        self._conditions = conditions
        super().__init__()

        if pyccel_stage != 'syntactic':
            self._shape      = lhs.shape
            self._class_type = lhs.class_type

    @property
    def loops(self):
        return self._loops

    @property
    def expr(self):
        return self._expr

    @property
    def lhs(self):
        return self._lhs

    @property
    def indices(self):
        return self._indices

    @property
    def index(self):
        return self._index

    @property
    def operations(self):
        """
        A dictionary mapping each type of comprehension to the operation used for populating it.

        For example, for list comprehensions we might use 
        ``{'list': 'append'}``, and for NumPy arrays, (which require a fixed size
        at compile time), we might use  ``{'numpy_array': [Assign_node]}``. 
        This mapping allows the code generator to select the appropriate operation 
        when building the final data structure from the comprehension.

        Returns
        -------
        dict
            A dictionary that maps comprehension types (e.g. 'list', 'numpy_array') to 
            the corresponding operation (append call, assign, etc.).
        """
        return self._operations

    @property
    def target_type(self):
        """
        The type of the result of the functional for.

        The type of the result of the functional for. This is useful at
        the syntactic stage to pass along the final type of the lhs (list/set/array/etc).
        """
        return self._target_type

    @property
    def conditions(self):
        """
        A list of filter conditions for each loop in the comprehension.

        If not `None`, each element of this list is either an `If` instance 
        that describes the filtering condition for that loop, or `None` if 
        no condition is applied. These conditions collectively determine
        whether each item produced by the loops is included in the final result.

        Returns
        -------
        list[If|None], or None
            The list of filter conditions for each for-loop in 
            the comprehension, or `None` if not specified.
        """
        return self._conditions

#==============================================================================
class GeneratorComprehension(FunctionalFor):
    """ Super class for all functions which reduce generator expressions to scalars
    """
    __slots__ = ()

#==============================================================================
class FunctionalSum(GeneratorComprehension):
    """ Represents a call to sum for a list argument
    >>> sum([i in range(5)])
    """
    __slots__ = ()
    name = 'sum'

#==============================================================================
class FunctionalMax(GeneratorComprehension):
    """ Represents a call to max for a list argument
    >>> max([i in range(5)])
    """
    __slots__ = ()
    name = 'max'
#==============================================================================

class FunctionalMin(GeneratorComprehension):
    """ Represents a call to min for a list argument
    >>> min([i in range(5)])
    """
    __slots__ = ()
    name = 'min'
#==============================================================================

class MaxLimit(TypedAstNode):
    """
    A class representing the largest usable value for a given type.

    A class representing the largest usable value for a given type.
    This is particularly useful with FunctionalMin.

    Parameters
    ----------
    class_type : FixedSizeNumericType
        The type whose maximum limit is represented by an instance
        of this class.
    """
    __slots__ = ('_class_type',)
    _attribute_nodes = ()

    def __init__(self, class_type):
        assert class_type.rank == 0
        self._class_type = class_type
        super().__init__()
#==============================================================================

class MinLimit(TypedAstNode):
    """
    A class representing the smallest usable value for a given type.

    A class representing the smallest usable value for a given type.
    This is particularly useful with FunctionalMax.

    Parameters
    ----------
    class_type : FixedSizeNumericType
        The type whose minimum limit is represented by an instance
        of this class.
    """
    __slots__ = ('_class_type',)
    _attribute_nodes = ()

    def __init__(self, class_type):
        assert isinstance(class_type, FixedSizeNumericType)
        self._class_type = class_type
        super().__init__()
