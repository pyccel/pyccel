# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
""" Module containing the Scope class
"""

from pyccel.ast.core      import ClassDef
from pyccel.ast.datatypes import DataTypeFactory
from pyccel.ast.headers   import MacroFunction, MacroVariable
from pyccel.ast.headers   import FunctionHeader, ClassHeader, MethodHeader
from pyccel.ast.internals import PyccelSymbol
from pyccel.ast.variable  import Variable, DottedName

from pyccel.errors.errors import Errors

from pyccel.utilities.strings import create_incremented_string

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
            '_is_loop','_loops','_temporary_variables', '_used_symbols',
            '_dummy_counter')

    categories = ('functions','variables','classes',
            'imports','symbolic_functions',
            'macros','templates','headers','decorators',
            'cls_constructs')

    def __init__(self, *, decorators=None, is_loop = False,
                    parent_scope = None):

        self._imports = {k:{} for k in self.categories}

        self._locals  = {k:{} for k in self.categories}

        self._temporary_variables = []

        self._used_symbols = set()

        self._dummy_counter = 0

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
        assert var.name!='_'
        if not isinstance(var, Variable):
            raise TypeError('variable must be of type Variable')

        if name is None:
            name = var.name

        if not allow_loop_scoping and self.is_loop:
            self.parent_scope.insert_variable(var, name, allow_loop_scoping)
        else:
            if name in self._locals['variables']:
                raise RuntimeError('New variable {} already exists in scope'.format(name))
            if name == '_':
                self._temporary_variables.append(var)
            else:
                self._locals['variables'][name] = var

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

    def insert_symbol(self, symbol):
        """ Add a new symbol to the scope
        """
        self._used_symbols.add(symbol)

    def insert_symbols(self, symbols):
        """ Add multiple new symbols to the scope
        """
        self._used_symbols.update(symbols)

    @property
    def all_used_symbols(self):
        if self.parent_scope:
            symbols = self.parent_scope.all_used_symbols
        else:
            symbols = set()
        symbols.update(self._used_symbols)
        return symbols

    @property
    def local_used_symbols(self):
        return self._used_symbols

    def get_new_incremented_symbol(self, prefix, counter):
        """
        Creates a new name by adding a numbered suffix to the provided prefix.

          Parameters
          ----------
          prefix : str

          Returns
          -------
          new_name     : str
        """

        new_name, counter = create_incremented_string(self.local_used_symbols, prefix = prefix)

        self.insert_symbol(new_name)

        return PyccelSymbol(new_name, is_temp=True), counter

    def get_new_name(self, current_name = None):
        """
        Creates a new name. A current_name can be provided indicating the name the
        user would like to use if possible. If this name is not available then it
        will be used as a prefix for the new name.
        If no current_name is provided, then the standard prefix is used, and the
        dummy counter is used and updated to facilitate finding the next value of
        this common case

          Parameters
          ----------
          current_name : str

          Returns
          -------
          new_name     : str
        """
        if current_name is not None and current_name not in self.local_used_symbols:
            self.insert_symbol(current_name)
            return current_name

        if current_name is None:
            # Avoid confusing names by also searching in parent scopes
            new_name, self._dummy_counter = create_incremented_string(self.all_used_symbols,
                                                prefix = current_name,
                                                counter = self._dummy_counter)
        else:
            # When a name is suggested, try to stick to it
            new_name,_ = create_incremented_string(self.local_used_symbols, prefix = current_name)

        self.insert_symbol(new_name)

        return new_name

    def get_new_symbol(self):
        return PyccelSymbol(self.get_new_name(), is_temp=True)

    def get_available_name(self, start_name):
        if start_name == '_':
            return self.get_new_name()
        elif self.is_loop:
            return self.parent_scope.get_available_name(start_name)
        elif start_name in self._used_symbols:
            return start_name
        elif start_name in self.parent_scope.all_used_symbols:
            return self.get_new_name(start_name)
        else:
            return start_name

    def create_product_loop_scope(self, inner_scope, n_loops):
        """ Create a n_loops loop scopes such that the innermost loop
        has the scope inner_scope

        Parameters
        ----------
        inner_scope : Namespace
                      Namespace describing the innermost scope
        n_loops     : The number of loop scopes required
        """
        assert inner_scope == self._loops[-1]
        self._loops.pop()
        scopes = [self.create_new_loop_scope()]
        for i in range(n_loops-2):
            scopes.append(scopes[-1].create_new_loop_scope())
        scopes[-1]._loops.append(inner_scope)
        inner_scope.parent_scope = scopes[-1]
        scopes.append(inner_scope)
        return scopes

    def collect_all_imports(self):
        """ Collect the names of all files necessary to understand this scope
        """
        imports = list(self._imports['imports'].keys())
        imports.extend([i for s in self._sons_scopes.values() for i in s.collect_all_imports()])
        return imports
