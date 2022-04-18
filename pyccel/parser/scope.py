# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
""" Module containing the Scope class
"""
from pyccel.ast.core import ClassDef
from pyccel.ast.datatypes import DataTypeFactory
from pyccel.ast.headers import MacroFunction, MacroVariable
from pyccel.ast.headers import FunctionHeader, ClassHeader, MethodHeader
from pyccel.ast.variable import Variable, DottedName

from pyccel.errors.errors import Errors

errors = Errors()

class Scope(object):
    """ Class representing all objects defined within a given
    scope

    Parameters
    ----------
    decorators : dict
                 A dictionary of any decorators which operate on
                 objects in this scope
    """
    __slots__ = ('_imports','_locals','parent_scope','_sons_scopes',
            '_is_loop','_loops','_temporary_variables')

    categories = ('functions','variables','classes',
            'imports','symbolic_functions',
            'macros','templates','headers','decorators',
            'cls_constructs')

    def __init__(self, *, decorators=None, is_loop = False,
                    parent_scope = None):

        self._imports = {k:{} for k in self.categories}

        self._locals  = {k:{} for k in self.categories}

        self._temporary_variables = []

        if decorators:
            self._locals['decorators'].update(decorators)

        # TODO use another name for headers
        #      => reserved keyword, or use __
        self.parent_scope        = parent_scope
        self._sons_scopes        = {}


        self._is_loop = is_loop
        # scoping for loops
        self._loops = []

    def new_child_scope(self, name, **kwargs):
        """
        Create a new child Scope object which has the current object as parent.

        The parent scope can access the child scope through the '_sons_scopes'
        dictionary, using the provided name as key. Conversely, the child scope
        can access the parent scope through the 'parent_scope' attribute.

        Parameters
        ----------
        name : str
            Name of the new scope, used as a key to retrieve the new scope.

        kwargs : dict
            Keyword arguments passed to __init__() for object initialization.

        Returns
        -------
        child : Scope
            New child scope, which has the current object as parent.

        """

        child = Scope(**kwargs, parent_scope = self)

        self._sons_scopes[name] = child
        child.parent_scope = self

        return child

    @property
    def imports(self):
        """ A dictionary of objects imported in this scope
        """
        return self._imports

    @property
    def variables(self):
        """ A dictionary of variables defined in this scope
        """
        return self._locals['variables']

    @property
    def classes(self):
        """ A dictionary of classes defined in this scope
        """
        return self._locals['classes']

    @property
    def functions(self):
        """ A dictionary of functions defined in this scope
        """
        return self._locals['functions']

    @property
    def macros(self):
        """ A dictionary of macros defined in this scope
        """
        return self._locals['macros']

    @property
    def headers(self):
        """A dictionary of user defined headers which may
        be applied to functions in this scope"""
        return self._locals['headers']

    @property
    def templates(self):
        """A dictionary of user defined templates which may
        be applied to functions in this scope"""
        return self._locals['templates']

    @property
    def decorators(self):
        """Dictionary of Pyccel decorators which may be
        applied to a function definition in this scope."""
        return self._locals['decorators']

    @property
    def cls_constructs(self):
        """ A dictionary of datatypes for the classes defined in
        this scope
        """
        return self._locals['cls_constructs']

    @property
    def sons_scopes(self):
        """ A dictionary of all the scopes contained within the
        current scope
        """
        return self._sons_scopes

    @property
    def symbolic_functions(self):
        """ A dictionary of symbolic functions defined in this scope
        """
        return self._locals['symbolic_functions']

    def find(self, name, category = None):
        """ Find and return the specified object in the scope.
        If the object cannot be found then None is returned

        Parameters
        ----------
        name : str
            The name of the object we are searching for
        category : str
            The type of object we are searching for.
            This must be one of the strings in Scope.categories
        """
        for l in ([category] if category else self._locals.keys()):
            if name in self._locals[l]:
                return self._locals[l][name]

            if name in self.imports[l]:
                return self.imports[l][name]

        # Walk up the tree of Scope objects, until the root if needed
        if self.parent_scope:
            return self.parent_scope.find(name, category)
        else:
            return None

    def find_all(self, category):
        """ Find and return all objects from the specified category
        in the scope.

        Parameter
        ---------
        category : str
            The type of object we are searching for.
            This must be one of the strings in Scope.categories
        """
        if self.parent_scope:
            result = self.parent_scope.find_all(category)
        else:
            result = {}

        result.update(self._locals[category])

        return result

    @property
    def is_loop(self):
        """ Indicates whether this scope describes a loop
        """
        return self._is_loop

    @property
    def loops(self):
        """ Returns the scopes associated with any loops within this scope
        """
        return self._loops

    def create_new_loop_scope(self):
        """ Create a new Scope within the current scope describing
        a loop (For/While/etc)
        """
        new_scope = Scope(decorators=self.decorators, is_loop = True,
                        parent_scope = self)
        self._loops.append(new_scope)
        return new_scope

    def insert_variable(self, var, name = None, allow_loop_scoping = False):
        """ Add a variable to the current scope

        Parameters
        ----------
        var  : Variable
                The variable to be inserted into the current scope
        name : str
                The name of the variable in the python code
                Default : var.name
        python_scope : bool
                If true then we assume that python scoping applies.
                In this case variables declared in loops exist beyond
                the end of the loop. Otherwise variables may be local
                to loops
                Default : True
        """
        if not isinstance(var, Variable):
            raise TypeError('variable must be of type Variable')

        if name is None:
            name = var.name

        if not allow_loop_scoping and self.is_loop:
            self.parent_scope.insert_variable(var, name, allow_loop_scoping)
        else:
            if name in self._locals['variables']:
                raise RuntimeError('New variable {} already exists in scope'.format(name))
            self._locals['variables'][name] = var
            self._temporary_variables.append(var)

    def remove_variable(self, var, name = None):
        """ Remove a variable from anywhere in scope

        Parameters
        ----------
        var  : Variable
                The variable to be removed
        name : str
                The name of the variable in the python code
                Default : var.name
        """
        if name is None:
            name = var.name

        if name in self._locals['variables']:
            self._locals['variables'].pop(name)
        elif self.parent_scope:
            self.parent_scope.remove_variable(var, name)
        else:
            raise RuntimeError("Variable not found in scope")

    def insert_class(self, cls):
        """ Add a class to the current scope

        Parameters
        ----------
        cls  : ClassDef
                The class to be inserted into the current scope
        """
        if not isinstance(cls, ClassDef):
            raise TypeError('class must be of type ClassDef')

        name = cls.name

        if self.is_loop:
            self.parent_scope.insert_class(cls)
        else:
            #if name in self._locals['classes']:
            #    raise RuntimeError('New class already exists in scope')
            self._locals['classes'][name] = cls

    def insert_macro(self, macro):
        """ Add a macro to the current scope
        """

        if not isinstance(macro, (MacroFunction, MacroVariable)):
            raise TypeError('Expected a macro')

        name = macro.name
        if isinstance(macro.name, DottedName):
            name = name.name[-1]

        self._locals['macros'][name] = macro

    def insert_template(self, expr):
        """append the scope's templates with the given template"""
        self._locals['templates'][expr.name] = expr

    def insert_header(self, expr):
        """ Add a header to the current scope
        """
        if isinstance(expr, (FunctionHeader, MethodHeader)):
            if expr.name in self.headers:
                self.headers[expr.name].append(expr)
            else:
                self.headers[expr.name] = [expr]
        elif isinstance(expr, ClassHeader):
            self.headers[expr.name] = expr

            #  create a new Datatype for the current class

            iterable = 'iterable' in expr.options
            with_construct = 'with' in expr.options
            dtype = DataTypeFactory(expr.name, '_name',
                                    is_iterable=iterable,
                                    is_with_construct=with_construct)
            self.cls_constructs[expr.name] = dtype
        else:
            msg = 'header of type{0} is not supported'
            msg = msg.format(str(type(expr)))
            raise TypeError(msg)
