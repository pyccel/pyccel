# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
""" Module containing the Scope class
"""

from pyccel.ast.core      import ClassDef
from pyccel.ast.headers   import MacroFunction, MacroVariable
from pyccel.ast.headers   import FunctionHeader, MethodHeader
from pyccel.ast.internals import PyccelSymbol
from pyccel.ast.variable  import Variable, DottedName

from pyccel.parser.syntax.headers import FunctionHeaderStmt

from pyccel.errors.errors import Errors

from pyccel.naming.pythonnameclashchecker import PythonNameClashChecker

from pyccel.utilities.strings import create_incremented_string

errors = Errors()

class Scope(object):
    """
    Class representing all objects defined within a given scope.

    This class provides all necessary functionalities for creating new object
    names without causing name clashes. It also stores all objects defined
    within the scope. This allows us to search for variables only in relevant
    scopes.

    Parameters
    ----------
    decorators : dict, default: ()
        A dictionary of any decorators which operate on objects in this scope.

    is_loop : bool, default: False
        Indicates if the scope represents a loop (in Python variables declared
        in loops are not scoped to the loop).

    parent_scope : Scope, default: None
        The enclosing scope.

    used_symbols : set, default: None
        A set of all the names which we know will appear in the scope and which
        we therefore want to avoid when creating new names.

    original_symbols : dict, default: None
        A dictionary which maps names used in the code to the original name used
        in the Python code.
    """
    allow_loop_scoping = False
    name_clash_checker = PythonNameClashChecker()
    __slots__ = ('_imports','_locals','_parent_scope','_sons_scopes',
            '_is_loop','_loops','_temporary_variables', '_used_symbols',
            '_dummy_counter','_original_symbol', '_dotted_symbols')

    categories = ('functions','variables','classes',
            'imports','symbolic_functions', 'symbolic_alias',
            'macros','templates','headers','decorators',
            'cls_constructs')

    def __init__(self, *, decorators = (), is_loop = False,
                    parent_scope = None, used_symbols = None,
                    original_symbols = None):

        self._imports = {k:{} for k in self.categories}

        self._locals  = {k:{} for k in self.categories}

        self._temporary_variables = []

        if used_symbols and not isinstance(used_symbols, dict):
            raise RuntimeError("Used symbols must be a dictionary")

        self._used_symbols = used_symbols or {}
        self._original_symbol = original_symbols or {}

        self._dummy_counter = 0

        self._locals['decorators'].update(decorators)

        # TODO use another name for headers
        #      => reserved keyword, or use __
        self._parent_scope       = parent_scope
        self._sons_scopes        = {}

        self._is_loop = is_loop
        # scoping for loops
        self._loops = []

        self._dotted_symbols = []

    def __setstate__(self, state):
        state = state[1] # Retrieve __dict__ ignoring None
        if any(s not in state for s in self.__slots__):
            raise AttributeError("Missing attribute from slots. Please update pickle file")

        for s in state:
            setattr(self, s, state[s])

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
        ps = kwargs.pop('parent_scope', self)
        if ps is not self:
            raise ValueError("A child of {} cannot have a parent {}".format(self, ps))

        child = Scope(**kwargs, parent_scope = self)

        self.add_son(name, child)

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
    def symbolic_alias(self):
        """
        A dictionary of symbolic alias defined in this scope.

        A symbolic alias is a symbol declared in the scope which is mapped
        to a constant object. E.g. a symbol which represents a type.
        """
        return self._locals['symbolic_alias']

    @property
    def symbolic_functions(self):
        """ A dictionary of symbolic functions defined in this scope
        """
        return self._locals['symbolic_functions']

    def find(self, name, category = None, local_only = False):
        """
        Find and return the specified object in the scope.

        Find a specified object in the scope and return it.
        The object is identified by a string contianing its name.
        If the object cannot be found then None is returned.

        Parameters
        ----------
        name : str
            The name of the object we are searching for.
        category : str, optional
            The type of object we are searching for.
            This must be one of the strings in Scope.categories.
            If no value is provided then we look in all categories.
        local_only : bool, default=False
            Indicates whether we should look for variables in the
            entire scope or whether we should limit ourselves to the
            local scope.
        """
        for l in ([category] if category else self._locals.keys()):
            if name in self._locals[l]:
                return self._locals[l][name]

            if name in self.imports[l]:
                return self.imports[l][name]

        # Walk up the tree of Scope objects, until the root if needed
        if self.parent_scope and (self.is_loop or not local_only):
            return self.parent_scope.find(name, category, local_only)
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
        self.add_loop(new_scope)
        return new_scope

    def insert_variable(self, var, name = None):
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
        if var.name == '_':
            raise ValueError("A temporary variable should have a name generated by Scope.get_new_name")
        if not isinstance(var, Variable):
            raise TypeError('variable must be of type Variable')

        if name is None:
            name = var.name

        if not self.allow_loop_scoping and self.is_loop:
            self.parent_scope.insert_variable(var, name)
        else:
            if name in self._locals['variables']:
                raise RuntimeError('New variable {} already exists in scope'.format(name))
            if name == '_':
                self._temporary_variables.append(var)
            else:
                self._locals['variables'][name] = var
            if name not in self.local_used_symbols.values():
                self.insert_symbol(name)

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

        self._used_symbols.pop(name)

        if name in self._locals['variables']:
            self._locals['variables'].pop(name)
        elif self.parent_scope:
            self.parent_scope.remove_variable(var, name)
        else:
            raise RuntimeError("Variable not found in scope")

    def insert_class(self, cls):
        """
        Add a class to the current scope.

        Add the definition of a class to the current scope to
        make it discoverable when used.

        Parameters
        ----------
        cls : ClassDef
            The class to be inserted into the current scope.
        """
        if not isinstance(cls, ClassDef):
            raise TypeError('class must be of type ClassDef')

        name = cls.name

        if self.is_loop:
            self.parent_scope.insert_class(cls)
        else:
            if name in self._locals['classes']:
                raise RuntimeError(f"A class with name '{name}' already exists in the scope")
            self._locals['classes'][name] = cls

    def update_class(self, cls):
        """
        Update a class which is in scope.

        Search for a class in the current scope and its parents. Once it
        has been found, replace it with the updated ClassDef passed as
        argument.

        Parameters
        ----------
        cls : ClassDef
            The class to be inserted into the current scope.
        """
        if not isinstance(cls, ClassDef):
            raise TypeError('class must be of type ClassDef')

        name = cls.name

        name_found = name in self._locals['classes']

        if not name_found and self.parent_scope:
            self.parent_scope.update_class(cls)
        else:
            if not name_found:
                raise RuntimeError('Class not found in scope')
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
        """
        Add a header to the current scope.

        Add a header describing a function, method or class to
        the current scope.

        Parameters
        ----------
        expr : pyccel.ast.Header
            The header description.

        Raises
        ------
        TypeError
            Raised if the header type is unknown.
        """
        if isinstance(expr, (FunctionHeader, MethodHeader, FunctionHeaderStmt)):
            if expr.name in self.headers:
                self.headers[expr.name].append(expr)
            else:
                self.headers[expr.name] = [expr]
        else:
            msg = 'header of type{0} is not supported'
            msg = msg.format(str(type(expr)))
            raise TypeError(msg)

    def insert_symbol(self, symbol):
        """ Add a new symbol to the scope
        """
        if isinstance(symbol, DottedName):
            self._dotted_symbols.append(symbol)
        else:
            if not self.allow_loop_scoping and self.is_loop:
                self.parent_scope.insert_symbol(symbol)
            elif symbol not in self._used_symbols:
                collisionless_symbol = self.name_clash_checker.get_collisionless_name(symbol,
                        self._used_symbols.values())
                collisionless_symbol = PyccelSymbol(collisionless_symbol,
                        is_temp = getattr(symbol, 'is_temp', False))
                self._used_symbols[symbol] = collisionless_symbol
                self._original_symbol[collisionless_symbol] = symbol

    def insert_symbolic_alias(self, symbol, alias):
        """
        Add a new symbolic alias to the scope.

        A symbolic alias is a symbol declared in the scope which is mapped
        to a constant object. E.g. a symbol which represents a type.
        """
        if symbol in self._locals['symbolic_alias']:
            errors.report(f"{symbol} cannot represent multiple static concepts",
                    symbol=symbol, severity='error')

        self._locals['symbolic_alias'][symbol] = alias

    def insert_symbols(self, symbols):
        """ Add multiple new symbols to the scope
        """
        for s in symbols:
            self.insert_symbol(s)

    @property
    def dotted_symbols(self):
        return self._dotted_symbols

    @property
    def all_used_symbols(self):
        """ Get all symbols which already exist in this scope
        """
        if self.parent_scope:
            symbols = self.parent_scope.all_used_symbols
        else:
            symbols = set()
        symbols.update(self._used_symbols.values())
        return symbols

    @property
    def local_used_symbols(self):
        """ Get all symbols which already exist in this scope
        excluding enclosing scopes
        """
        return self._used_symbols

    def get_new_incremented_symbol(self, prefix, counter):
        """
        Create a new name by adding a numbered suffix to the provided prefix.

        Create a new name which does not clash with any existing names by
        adding a numbered suffix to the provided prefix.

        Parameters
        ----------
        prefix : str
            The prefix from which the new name will be created.

        counter : int
            The starting point for the incrementation.

        Returns
        -------
        pyccel.ast.internals.PyccelSymbol
            The newly created name.
        """

        new_name, counter = create_incremented_string(self.local_used_symbols.values(),
                                    prefix = prefix, counter = counter, name_clash_checker = self.name_clash_checker)

        new_symbol = PyccelSymbol(new_name, is_temp=True)

        self.insert_symbol(new_symbol)

        return new_symbol, counter

    def get_new_name(self, current_name = None):
        """
        Get a new name which does not clash with any names in the current context.

        Creates a new name. A current_name can be provided indicating the name the
        user would like to use if possible. If this name is not available then it
        will be used as a prefix for the new name.
        If no current_name is provided, then the standard prefix is used, and the
        dummy counter is used and updated to facilitate finding the next value of
        this common case.

        Parameters
        ----------
        current_name : str, default: None
            The name the user would like to use if possible.

        Returns
        -------
        PyccelSymbol
            The new name which will be printed in the code.
        """
        if current_name is not None and not self.name_clash_checker.has_clash(current_name, self.all_used_symbols):
            new_name = PyccelSymbol(current_name)
            self.insert_symbol(new_name)
            return new_name

        if current_name is None:
            # Avoid confusing names by also searching in parent scopes
            new_name, self._dummy_counter = create_incremented_string(self.all_used_symbols,
                                                prefix = current_name,
                                                counter = self._dummy_counter,
                                                name_clash_checker = self.name_clash_checker)
        else:
            # When a name is suggested, try to stick to it
            new_name,_ = create_incremented_string(self.all_used_symbols, prefix = current_name)

        new_name = PyccelSymbol(new_name, is_temp = True)
        self.insert_symbol(new_name)

        return new_name

    def get_temporary_variable(self, dtype_or_var, name = None, **kwargs):
        """
        Get a temporary variable

        Parameters
        ----------
        dtype_or_var : str, DataType, Variable
            In the case of a string of DataType: The type of the Variable to be created
            In the case of a Variable: a Variable which will be cloned to set all the Variable properties
        name : str
            The requested name for the new variable
        kwargs : dict
            See Variable keyword arguments
        """
        assert isinstance(name, (str, type(None)))
        name = self.get_new_name(name)
        if isinstance(dtype_or_var, Variable):
            var = dtype_or_var.clone(name, **kwargs, is_temp = True)
        else:
            var = Variable(dtype_or_var, name, **kwargs, is_temp = True)
        self.insert_variable(var)
        return var

    def get_expected_name(self, start_name):
        """ Get a name with no collisions, ideally the provided name.
        The provided name should already exist in the symbols
        """
        if start_name == '_':
            return self.get_new_name()
        elif start_name in self._used_symbols.keys():
            return self._used_symbols[start_name]
        elif self.parent_scope:
            return self.parent_scope.get_expected_name(start_name)
        else:
            raise RuntimeError("{} does not exist in scope".format(start_name))

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
        scopes = [self.create_new_loop_scope()]
        for _ in range(n_loops-2):
            scopes.append(scopes[-1].create_new_loop_scope())
        inner_scope.update_parent_scope(scopes[-1], is_loop = True)
        scopes.append(inner_scope)
        return scopes

    def collect_all_imports(self):
        """ Collect the names of all modules necessary to understand this scope
        """
        imports = list(self._imports['imports'].keys())
        imports.extend([i for s in self._sons_scopes.values() for i in s.collect_all_imports()])
        return imports

    def update_parent_scope(self, new_parent, is_loop, name = None):
        """ Change the parent scope
        """
        if is_loop:
            if self.parent_scope:
                self.parent_scope.remove_loop(self)
            self._parent_scope = new_parent
            self.parent_scope.add_loop(self)
        else:
            if self.parent_scope:
                name = self.parent_scope.remove_son(self)
            self._parent_scope = new_parent
            self.parent_scope.add_son(name, self)

    @property
    def parent_scope(self):
        """ Return the enclosing scope
        """
        return self._parent_scope

    def remove_loop(self, loop):
        """ Remove a loop from the scope
        """
        self._loops.remove(loop)

    def remove_son(self, son):
        """ Remove a sub-scope from the scope
        """
        name = [k for k,v in self._sons_scopes.items() if v is son]
        assert len(name) == 1
        self._sons_scopes.pop(name[0])

    def add_loop(self, loop):
        """ Make parent aware of new child loop
        """
        assert loop.parent_scope is self
        self._loops.append(loop)

    def add_son(self, name, son):
        """ Make parent aware of new child
        """
        assert son.parent_scope is self
        self._sons_scopes[name] = son

    def get_python_name(self, name):
        """ Get the name used in the original python code from the
        name used by the variable
        """
        if name in self._original_symbol:
            return self._original_symbol[name]
        elif self.parent_scope:
            return self.parent_scope.get_python_name(name)
        else:
            raise RuntimeError("Can't find {} in scope".format(name))

    @property
    def python_names(self):
        """ Get map of new names to original python names
        """
        return self._original_symbol
