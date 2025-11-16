#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
""" Module containing objects from the typing module understood by pyccel
"""
from immutabledict import immutabledict

from .basic     import TypedAstNode
from .core      import Module, PyccelFunctionDef
from .datatypes import TypeAlias, GenericType, FinalType

__all__ = (
    'TypingAnnotation',
    'TypingAny',
    'TypingFinal',
    'TypingOverload',
    'TypingTypeAlias',
    'TypingTypeVar',
    'typing_mod'
)

#==============================================================================

class TypingFinal(TypedAstNode):
    """
    Class representing a call to the typing.Final construct.

    Class representing a call to the typing.Final construct. A "call" to this
    object looks like an IndexedElement. This is because types are involved.

    Parameters
    ----------
    arg : SyntacticTypeAnnotation
        The annotation which is coerced to be constant.
    """
    __slots__ = ('_arg',)
    _attribute_nodes = ('_arg',)
    name = 'Final'
    _static_type = FinalType

    def __init__(self, arg):
        self._arg = arg
        super().__init__()

    @property
    def arg(self):
        """
        Get the argument describing the type annotation for an object.

        Get the argument describing the type annotation for an object.
        """
        return self._arg

#==============================================================================

class TypingAnnotation(TypedAstNode):
    """
    Class representing a call to the typing.Annotated construct.

    Class representing a call to the typing.Annotated construct. A "call" to this
    object looks like an IndexedElement. This is because types are involved. It
    allows 

    Parameters
    ----------
    arg : SyntacticTypeAnnotation
        The annotation which is annotated.
    **metadata
        The metadata providing additional information about the variable being
        declared.
    """
    __slots__ = ('_arg','_metadata')
    _attribute_nodes = ('_arg',)
    name = 'Annotated'

    def __init__(self, arg, **metadata):
        self._arg = arg
        self._metadata = metadata
        super().__init__()

    @property
    def arg(self):
        """
        Get the argument describing the type annotation for an object.

        Get the argument describing the type annotation for an object.
        """
        return self._arg

    @property
    def metadata(self):
        """
        The metadata providing additional information about the variable being declared.

        The metadata providing additional information about the variable being declared.
        """
        return immutabledict(self._metadata)

#==============================================================================
class TypingTypeAlias(TypedAstNode):
    """
    Class representing a call to the typing.TypeAlias construct.

    Class representing a call to the typing.TypeAlias construct. This object
    is only used for type annotations. It is useful for creating a PyccelFunctionDef
    but instances should not be created.
    """
    __slots__ = ()
    _attribute_nodes = ()
    _static_type = TypeAlias()

#==============================================================================
class TypingTypeVar(TypedAstNode):
    """
    Class representing a call to the typing.TypeVar construct.

    Class representing a call to the typing.TypeVar construct. This object
    is a type annotation.

    Parameters
    ----------
    name : str
        The name which will be used to identify the TypeVar.
    *constraints : PyccelAstNode
        The possible annotations that this TypeVar can represent.
    bound : PyccelAstNode
        The superclass from which the type must inherit. See PEP 484.
    covariant : bool
        Indicates if the TypeVar can represent superclasses of the constraints. See PEP 484.
    contravariant : bool
        Indicates if the TypeVar can represent subclasses of the constraints. See PEP 484.
    infer_variance : bool
        Indicates if the variance (see covariant/contravariant) should be inferred from
        use. See PEP 695.
    default : TypedAstNode
        The type that should be chosen if the type cannot be deduced from the call.
        This can sometimes be the case for parametrised classes. See PEP 696.
    """
    __slots__ = ('_name', '_possible_types', '_default')
    _attribute_nodes = ()
    _class_type = TypeAlias()
    _shape = None
    name = 'TypeVar'

    def __init__(self, name, *constraints, bound=None, covariant=False, contravariant=False,
            infer_variance=False, default=None):
        if covariant or contravariant or bound:
            raise TypeError("Covariant, contravariant and bound types are not currently supported")
        if len(constraints) == 0:
            raise TypeError(f"The possible types for {name} must be specified")
        if default is not None and default not in constraints:
            raise TypeError("The default value of the TypeVar must be one of the constraints.")
        self._name = name
        self._possible_types = constraints
        self._default = default
        super().__init__()

    @property
    def name_str(self):
        """
        The name that is printed to represent the TypeVar.

        The name that is printed to represent the TypeVar.
        """
        return self._name

    @property
    def type_list(self):
        """
        Get the list of types which this TypeVar can take.

        Get the list of types which this TypeVar can take (stored in a tuple).
        """
        return self._possible_types

#==============================================================================
class TypingOverload(TypedAstNode):
    """
    Class representing a call to the typing.overload decorator.

    Class representing a call to the typing.overload decorator. This object
    will never be constructed. It exists to recognise the import.
    """
    __slots__ = ()
    _attribute_nodes = ()

#==============================================================================
class TypingAny(TypedAstNode):
    """
    Class representing a call to the typing.Any construct.

    Class representing a call to the typing.Any construct. This object
    will never be constructed. It exists to recognise the import.
    """
    __slots__ = ()
    _attribute_nodes = ()
    _static_type = GenericType()

#==============================================================================

typing_funcs = {
        'Any': PyccelFunctionDef('Any', TypingAny),
        'Annotated': PyccelFunctionDef('Annotated', TypingAnnotation),
        'Final': PyccelFunctionDef('Final', TypingFinal),
        'TypeAlias': PyccelFunctionDef('TypeAlias', TypingTypeAlias),
        'TypeVar' : PyccelFunctionDef('TypeVar', TypingTypeVar),
        'overload': PyccelFunctionDef('overload', TypingOverload)
    }

typing_mod = Module('typing',
    variables = (),
    funcs = typing_funcs.values(),
    )
