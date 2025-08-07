# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
""" Module containing the Scope class
"""

from pyccel.ast.bind_c    import BindCVariable
from pyccel.ast.core      import ClassDef, FunctionDef
from pyccel.ast.datatypes import InhomogeneousTupleType
from pyccel.ast.internals import PyccelSymbol, PyccelFunction
from pyccel.ast.typingext import TypingTypeVar
from pyccel.ast.variable  import Variable, DottedName, AnnotatedPyccelSymbol
from pyccel.ast.variable  import IndexedElement, DottedVariable

from pyccel.errors.errors import Errors

from pyccel.naming.pythonnameclashchecker import PythonNameClashChecker

from pyccel.utilities.strings import create_incremented_string
from pyccel.utilities.tools import ReadOnlyDict

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
    name : str, optional
        The name of the scope. The value needs to be provided when it is not a loop.

    decorators : dict, default: ()
        A dictionary of any decorators which operate on objects in this scope.

    is_loop : bool, default: False
        Indicates if the scope represents a loop (in Python variables declared
        in loops are not scoped to the loop).

    parent_scope : Scope, default: None
        The enclosing scope.

    used_symbols : dict, default: None
        A dictionary mapping all the names which we know will appear in the scope and which
        we therefore want to avoid when creating new names to their collisionless name.

    original_symbols : dict, default: None
        A dictionary which maps names used in the code to the original name used
        in the Python code.

    symbolic_aliases : dict, optional
        A dictionary which maps indexed tuple elements to variables representing those
        elements. This argument should only be used after the semantic stage.
    """
    allow_loop_scoping = False
    name_clash_checker = PythonNameClashChecker()
    __slots__ = ('_name', '_imports','_locals','_parent_scope','_sons_scopes',
            '_is_loop','_loops','_temporary_variables', '_used_symbols',
            '_dummy_counter','_original_symbol', '_dotted_symbols')

    categories = ('functions','variables','classes',
            'imports', 'symbolic_aliases',
            'decorators', 'cls_constructs')

    def __init__(self, *, name=None, decorators = (), is_loop = False,
                    parent_scope = None, used_symbols = None,
                    original_symbols = None, symbolic_aliases = None):

        self._name    = name
        self._imports = {k:{} for k in self.categories}

        self._locals  = {k:{} for k in self.categories}

        self._temporary_variables = []

        if used_symbols and not isinstance(used_symbols, dict):
            raise RuntimeError("Used symbols must be a dictionary")

        self._used_symbols = used_symbols or {}
        self._original_symbol = original_symbols or {}

        self._dummy_counter = 0

        self._locals['decorators'].update(decorators)
        if symbolic_aliases:
            self._locals['symbolic_aliases'].update(symbolic_aliases)

        # TODO use another name for headers
        #      => reserved keyword, or use __
        self._parent_scope       = parent_scope
        self._sons_scopes        = {}

        self._is_loop = is_loop
        # scoping for loops
        self._loops = []

        self._dotted_symbols = []

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
        **kwargs : dict
            Keyword arguments passed to __init__() for object initialization.

        Returns
        -------
        Scope
            New child scope, which has the current object as parent.
        """
        ps = kwargs.pop('parent_scope', self)
        if ps is not self:
            raise ValueError(f"A child of {self} cannot have a parent {ps}")

        child = Scope(name=name, **kwargs, parent_scope = self)

        self.add_son(name, child)

        return child

    @property
    def name(self):
        """
        The name of the scope.

        The name of the scope.
        """
        return self._name

    @property
    def imports(self):
        """ A dictionary of objects imported in this scope
        """
        return self._imports

    @property
    def variables(self):
        """
        A dictionary of variables defined in this scope.

        A dictionary whose keys are the original Python names of the variables
        in the scope and whose values are Variable objects. When handling an
        inlined function it is possible that some of the values will not be
        Variable objects but rather the value that the variable takes in this
        context.
        """
        return ReadOnlyDict(self._locals['variables'])

    @property
    def classes(self):
        """
        A dictionary of classes defined in this scope.

        A dictionary whose keys are the original Python names of the classes
        in the scope and whose variables are ClassDef objects.
        """
        return ReadOnlyDict(self._locals['classes'])

    @property
    def functions(self):
        """
        A dictionary of functions defined in this scope.

        A dictionary whose keys are the original Python names of the functions
        in the scope and whose variables are ClassDef objects.
        """
        return ReadOnlyDict(self._locals['functions'])

    @property
    def decorators(self):
        """
        A dictionary of the decorators applied to the current function.

        A dictionary of the decorators which are applied to the function definition
        in this scope. The keys are the name of the decorator function. The values
        depend on the decorator.
        """
        return ReadOnlyDict(self._locals['decorators'])

    @property
    def cls_constructs(self):
        """
        A dictionary of datatypes for the classes defined in this scope.

        A dictionary whose keys are the original Python names of the classes
        found in this scope and whose values are the types inheriting from
        PyccelType which identify these classes.
        """
        return ReadOnlyDict(self._locals['cls_constructs'])

    @property
    def sons_scopes(self):
        """ A dictionary of all the scopes contained within the
        current scope
        """
        return self._sons_scopes

    @property
    def symbolic_aliases(self):
        """
        A dictionary of symbolic alias defined in this scope.

        A symbolic alias is a symbol declared in the scope which is mapped
        to a constant object. E.g. a symbol which represents a type.
        """
        return ReadOnlyDict(self._locals['symbolic_aliases'])

    def find(self, name, category = None, local_only = False, raise_if_missing = False):
        """
        Find and return the specified object in the scope.

        Find a specified object in the scope and return it.
        The object is identified by a string containing its name.
        If the object cannot be found then None is returned unless
        an error is requested.

        Parameters
        ----------
        name : str
            The Python name of the object we are searching for.
        category : str, optional
            The type of object we are searching for.
            This must be one of the strings in Scope.categories.
            If no value is provided then we look in all categories.
        local_only : bool, default=False
            Indicates whether we should look for variables in the
            entire scope or whether we should limit ourselves to the
            local scope.
        raise_if_missing : bool, default=False
            Indicates whether an error should be raised if the object
            cannot be found.

        Returns
        -------
        pyccel.ast.basic.PyccelAstNode
            The object stored in the scope.
        """
        for l in ([category] if category else self._locals.keys()):
            if name in self._locals[l]:
                return self._locals[l][name]

            if name in self.imports[l]:
                return self.imports[l][name]

        # Walk up the tree of Scope objects, until the root if needed
        if self.parent_scope and (self.is_loop or not local_only):
            return self.parent_scope.find(name, category, local_only, raise_if_missing)
        elif raise_if_missing:
            raise RuntimeError(f"Can't find expected object {name} in scope")
        else:
            return None

    def find_all(self, category):
        """
        Find and return all objects from the specified category in the scope.

        Find and return all objects from the specified category in the scope.

        Parameters
        ----------
        category : str
            The type of object we are searching for.
            This must be one of the strings in Scope.categories.

        Returns
        -------
        dict
            A dictionary containing all the objects of the specified category
            found in the scope.
        """
        if self.parent_scope:
            result = self.parent_scope.find_all(category)
        else:
            result = {}

        result.update(self._locals[category])
        result.update(self._imports[category])

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

    def insert_variable(self, var, name = None, tuple_recursive = True):
        """
        Add a variable to the current scope.

        Add a variable to the current scope.

        Parameters
        ----------
        var : Variable
            The variable to be inserted into the current scope.
        name : str, default=var.name
            The name of the variable in the Python code.
        tuple_recursive : bool, default=True
            Indicate whether inhomogeneous tuples should be inserted recursively.
            Generally this should be the case, but occasionally inhomogeneous tuples
            are created with pre-existent elements. In this case trying to insert
            these elements would create an error.
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
                if name in self.symbolic_aliases.values():
                    # If the syntactic name is in the symbolic aliases then the link was created
                    # at the syntactic stage. In this case the element will be created before the
                    # tuple
                    return
                else:
                    raise RuntimeError(f'New variable {name} already exists in scope')

            if isinstance(var.class_type, InhomogeneousTupleType) and tuple_recursive:
                for v in var:
                    self.insert_variable(self.collect_tuple_element(v))

            if name == '_':
                self._temporary_variables.append(var)
            else:
                self._locals['variables'][name] = var
            if name not in self.local_used_symbols.values():
                self.insert_symbol(name)

    def remove_variable(self, var, name = None):
        """
        Remove a variable from anywhere in scope.

        Remove a variable from anywhere in scope.

        Parameters
        ----------
        var : Variable
                The variable to be removed.
        name : str, optional
                The name of the variable in the python code
                Default : var.name.
        """
        if name is None:
            name = self.get_python_name(var.name)

        if name in self._locals['variables']:
            self._locals['variables'].pop(name)
            self._used_symbols.pop(name)
        elif self.parent_scope:
            self.parent_scope.remove_variable(var, name)
        else:
            raise RuntimeError("Variable not found in scope")

    def inline_variable_definition(self, var_value, name):
        """
        Add the definition of a variable inline.

        Add an object to the variables dictionary. This object will
        be returned when the variable is collected but may not be
        itself a variable. This is important when translating inlined
        functions. To ensure that when searching for the variables
        representing the arguments, the value is used directly.

        Parameters
        ----------
        var_value : TypedAstNode
            The value of the variable.
        name : str
            The name of the variable.
        """
        self._locals['variables'][name] = var_value
        self._used_symbols[name] = name

    def insert_class(self, cls, name = None):
        """
        Add a class to the current scope.

        Add the definition of a class to the current scope to
        make it discoverable when used.

        Parameters
        ----------
        cls : ClassDef
            The class to be inserted into the current scope.

        name : str, optional
            The name under which the classes should be indexed in the scope.
            This defaults to the name of the class in Python.
        """
        if not isinstance(cls, ClassDef):
            raise TypeError('class must be of type ClassDef')

        if self.is_loop:
            self.parent_scope.insert_class(cls, name)
        else:
            if name is None:
                name = cls.name
                assert cls.pyccel_staging == 'syntactic'
            if name in self._locals['classes']:
                raise RuntimeError(f"A class with name '{name}' already exists in the scope")
            assert name in self._used_symbols
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
        if cls.pyccel_staging != 'syntactic':
            name = self.get_python_name(name)

        name_found = name in self._locals['classes']

        if not name_found and self.parent_scope:
            self.parent_scope.update_class(cls)
        else:
            if not name_found:
                raise RuntimeError('Class not found in scope')
            self._locals['classes'][name] = cls

    def insert_cls_construct(self, class_type):
        """
        Add a class construct to the scope.

        Add a class construct to the scope. A class construct is a type inheriting from
        PyccelType which describes the type of a class.

        Parameters
        ----------
        class_type : PyccelType
            The construct to be inserted.
        """
        name = class_type.name
        assert name in self._used_symbols
        self._locals['cls_constructs'][name] = class_type

    def insert_function(self, func, name):
        """
        Add a function to the scope.

        Add a function to the scope. The key will be the original name of the
        function in the Python code.

        Parameters
        ----------
        func : FunctionDef
            The function to be inserted.
        name : str | PyccelSymbol
            The original name of the function in the Python code. This will be
            used as the key for the function in the scope.
        """
        assert name in self._used_symbols
        assert name not in self._locals['functions']
        self._locals['functions'][name] = func

    def remove_function(self, name):
        """
        Remove a function from the scope.

        Remove a function from the scope. This method is often used when handling
        Interfaces.

        Parameters
        ----------
        name : str
            The original name of the function in the Python code.
        """
        self._locals['functions'].pop(name)

    def insert_symbol(self, symbol):
        """
        Add a new symbol to the scope.

        Add a new symbol to the scope in the syntactic stage. This should be used to
        declare symbols defined by the user. Once the symbol is declared the Scope
        generates a collisionless name if necessary which can be used in the target
        language without causing problems by being a keyword or being confused with
        other symbols (e.g. in Fortran which is not case-sensitive). This new name
        can be retrieved later using `Scope.get_expected_name`.

        Parameters
        ----------
        symbol : PyccelSymbol | AnnotatedPyccelSymbol | DottedName
            The symbol to be added to the scope.
        """
        if isinstance(symbol, AnnotatedPyccelSymbol):
            symbol = symbol.name

        if isinstance(symbol, DottedName):
            self._dotted_symbols.append(symbol)
        else:
            if not self.allow_loop_scoping and self.is_loop:
                self.parent_scope.insert_symbol(symbol)
            elif symbol not in self._used_symbols:
                collisionless_symbol = self.name_clash_checker.get_collisionless_name(symbol,
                        self.all_used_symbols)
                collisionless_symbol = PyccelSymbol(collisionless_symbol,
                        is_temp = getattr(symbol, 'is_temp', False))
                self._used_symbols[symbol] = collisionless_symbol
                self._original_symbol[collisionless_symbol] = symbol

    def remove_symbol(self, symbol):
        """
        Remove symbol from the scope.

        Remove symbol from the scope.

        Parameters
        ----------
        symbol : PyccelSymbol
            The symbol to be removed from the scope.
        """

        if symbol in self._used_symbols:
            collisionless_symbol = self._used_symbols.pop(symbol)
            self._original_symbol.pop(collisionless_symbol)


    def insert_symbolic_alias(self, symbol, alias):
        """
        Add a new symbolic alias to the scope.

        A symbolic alias is a symbol declared in the scope which is mapped
        to a constant object. E.g. a symbol which represents a type.

        Parameters
        ----------
        symbol : PyccelSymbol
            The symbol which will represent the object in the code.
        alias : pyccel.ast.basic.Basic
            The object which will be represented by the symbol.
        """
        if not self.allow_loop_scoping and self.is_loop:
            self.parent_scope.insert_symbolic_alias(symbol, alias)
        else:
            symbolic_aliases = self._locals['symbolic_aliases']
            if symbol in symbolic_aliases:
                errors.report(f"{symbol} cannot represent multiple static concepts",
                        symbol=symbol, severity='error')

            symbolic_aliases[symbol] = alias

    def insert_symbols(self, symbols):
        """ Add multiple new symbols to the scope
        """
        for s in symbols:
            self.insert_symbol(s)

    @property
    def dotted_symbols(self):
        """
        Return all dotted symbols that were inserted into the scope.

        Return all dotted symbols that were inserted into the scope.
        This is useful to ensure that class variable names are
        in the class scope.
        """
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

    def symbol_in_use(self, name):
        """
        Determine if a name is already in use in this scope.

        Determine if a name is already in use in this scope.

        Parameters
        ----------
        name : PyccelSymbol
            The name we are searching for.

        Returns
        -------
        bool
            True if the name has already been inserted into this scope, False otherwise.
        """
        if name in self._used_symbols:
            return True
        elif self.parent_scope:
            return self.parent_scope.symbol_in_use(name)
        else:
            return False

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

        chosen_new_symbol = PyccelSymbol(new_name, is_temp=True)

        self.insert_symbol(chosen_new_symbol)

        # The symbol may be different to the one chosen in the case of collisions with language-specific terms)
        new_symbol = self._used_symbols[chosen_new_symbol]

        return new_symbol, counter

    def get_new_name(self, current_name = None, is_temp = None):
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

        is_temp : bool, optional
            Indicates if the generated symbol should be a temporary (i.e. an extra
            temporary object generated by Pyccel). This is always the case if no
            current_name is provided.

        Returns
        -------
        PyccelSymbol
            The new name which will be printed in the code.
        """
        if current_name is not None and not self.name_clash_checker.has_clash(current_name, self.all_used_symbols):
            new_name = PyccelSymbol(current_name, is_temp = is_temp)
            self.insert_symbol(new_name)
            return new_name

        if current_name is None:
            assert is_temp is None
            is_temp = True
            # Avoid confusing names by also searching in parent scopes
            new_name, self._dummy_counter = create_incremented_string(self.all_used_symbols,
                                                prefix = current_name,
                                                counter = self._dummy_counter,
                                                name_clash_checker = self.name_clash_checker)
        else:
            if is_temp is None:
                is_temp = True
            # When a name is suggested, try to stick to it
            new_name,_ = create_incremented_string(self.all_used_symbols, prefix = current_name)

        new_name = PyccelSymbol(new_name, is_temp = is_temp)
        self.insert_symbol(new_name)

        return new_name

    def get_temporary_variable(self, dtype_or_var, name = None, *, clone_scope = None, **kwargs):
        """
        Get a temporary variable.

        Get a temporary variable.

        Parameters
        ----------
        dtype_or_var : str, DataType, Variable
            In the case of a string of DataType: The type of the Variable to be created
            In the case of a Variable: a Variable which will be cloned to set all the Variable properties.
        name : str, optional
            The requested name for the new variable.
        clone_scope : Scope, optional
            A scope which can be used to look for tuple elements when cloning a
            Variable.
        **kwargs : dict
            See Variable keyword arguments.

        Returns
        -------
        Variable
            The temporary variable.
        """
        assert isinstance(name, (str, type(None)))
        name = self.get_new_name(name)
        if isinstance(dtype_or_var, Variable):
            var = dtype_or_var.clone(name, **kwargs, is_temp = True)
        else:
            var = Variable(dtype_or_var, name, **kwargs, is_temp = True)
        if isinstance(var.class_type, InhomogeneousTupleType):
            assert isinstance(dtype_or_var, Variable)
            assert clone_scope is not None
            for orig_vi, vi_idx in zip(dtype_or_var, var):
                vi = self.get_temporary_variable(clone_scope.collect_tuple_element(orig_vi), clone_scope = clone_scope)
                self.insert_symbolic_alias(vi_idx, vi)
        self.insert_variable(var, tuple_recursive = False)
        return var

    def get_expected_name(self, start_name):
        """
        Get a name with no collisions.

        Get a name with no collisions, ideally the provided name.
        The provided name should already exist in the symbols.

        Parameters
        ----------
        start_name : str
            The name which was used in the Python code.

        Returns
        -------
        PyccelSymbol
            The name which will be used in the generated code.
        """
        if start_name == '_':
            return self.get_new_name()
        elif start_name in self._used_symbols.keys():
            return self._used_symbols[start_name]
        elif self.parent_scope:
            return self.parent_scope.get_expected_name(start_name)
        else:
            raise RuntimeError(f"{start_name} does not exist in scope")

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

    def collect_all_type_vars(self):
        """
        Collect all TypeVar objects which are available in this scope.

        Collect all TypeVar objects which are available in this scope. This includes
        TypeVars declared in parent scopes.

        Returns
        -------
        list[TypeVar]
            A list of TypeVars in the scope.
        """
        type_vars = {n:t for n,t in self.symbolic_aliases.items() if isinstance(t, TypingTypeVar)}
        if self.parent_scope:
            parent_type_vars = self.parent_scope.collect_all_type_vars()
            parent_type_vars.update(type_vars)
            return parent_type_vars
        else:
            return type_vars

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
        """
        Get the name used in the original Python code.

        Get the name used in the original Python code from the name used
        by the variable that was created in the parser.

        Parameters
        ----------
        name : PyccelSymbol | str
            The name of the Variable in the generated code.

        Returns
        -------
        str
            The name of the Variable in the original code.
        """
        if name in self._original_symbol:
            return self._original_symbol[name]
        elif self.parent_scope:
            return self.parent_scope.get_python_name(name)
        else:
            raise RuntimeError(f"Can't find {name} in scope")

    @property
    def python_names(self):
        """ Get map of new names to original python names
        """
        return self._original_symbol

    def rename_function(self, o, name):
        """
        Rename a function that exists in the scope.

        Rename a function that exists in the scope. This is done by
        finding a new collisionless name, renaming the FunctionDef
        instance, and updating the dictionary of symbols.

        Parameters
        ----------
        o : FunctionDef
            The object that should be renamed.

        name : str
            The suggested name for the new function.
        """
        assert isinstance(o, FunctionDef)
        newname = self.get_new_name(name)
        python_name = self._original_symbol.pop(o.name)
        assert python_name == o.scope.python_names.pop(o.name)
        o.rename(newname)
        self._original_symbol[newname] = python_name
        o.scope.python_names[newname] = python_name

    def collect_tuple_element(self, tuple_elem):
        """
        Get an element of a tuple.

        This function is mainly designed to handle inhomogeneous tuples. Such tuples
        cannot be directly represented in low-level languages. Instead they are replaced
        by multiple variables representing each of the elements of the tuple. This
        function maps tuple elements (e.g. `var[0]`) to the variable representing that
        element in the low-level language (e.g. `var_0`).

        Parameters
        ----------
        tuple_elem : PyccelAstNode
            The element of the tuple obtained via the `__getitem__` function.

        Returns
        -------
        Variable
            The variable which represents the tuple element in a low-level language.

        Raises
        ------
        PyccelError
            An error is raised if the tuple element has not yet been added to the scope.
        """
        if isinstance(tuple_elem, IndexedElement) and isinstance(tuple_elem.base, DottedVariable):
            cls_scope = tuple_elem.base.lhs.cls_base.scope
            if cls_scope is not self:
                return cls_scope.collect_tuple_element(tuple_elem)

        if isinstance(tuple_elem, IndexedElement) and isinstance(tuple_elem.base.class_type, InhomogeneousTupleType) \
                and not isinstance(tuple_elem.base, PyccelFunction):
            if isinstance(tuple_elem.base, DottedVariable):
                class_var = tuple_elem.base.lhs
                base = tuple_elem.base.clone(tuple_elem.base.name, Variable)
                tuple_elem_search = IndexedElement(base, *tuple_elem.indices)
            else:
                class_var = None
                tuple_elem_search = tuple_elem

            result = self.find(tuple_elem_search, 'symbolic_aliases')

            if result is None:
                msg = f'Internal error. Tuple element {tuple_elem} could not be found.'
                return errors.report(msg,
                        symbol = tuple_elem,
                        severity='fatal')
            elif class_var:
                return result.clone(result.name, DottedVariable, lhs=class_var)
            else:
                return result
        else:
            return tuple_elem

    def collect_all_tuple_elements(self, tuple_var):
        """
        Create a tuple of variables from a variable representing an inhomogeneous object.

        Create a tuple of variables that can be printed in a low-level language. An
        inhomogeneous object cannot be represented as is in a low-level language so
        it must be unpacked into a PythonTuple. This function is recursive so that
        variables with a type such as `tuple[tuple[int,bool],float]` generate
        `PythonTuple(PythonTuple(var_0_0, var_0_1), var_1)`.

        Parameters
        ----------
        tuple_var : Variable | FunctionAddress
            A variable which may or may not be an inhomogeneous tuple.

        Returns
        -------
        list[Variable]
            All variables that should be printed in a low-level language to represent
            the Variable.
        """
        if isinstance(tuple_var, BindCVariable):
            tuple_var = tuple_var.new_var

        # A tuple_var may not be a Variable if we are collecting arguments.
        # In this case it may be something else, e.g. a FunctionAddress.
        if isinstance(tuple_var, Variable) and isinstance(tuple_var.class_type, InhomogeneousTupleType):
            return [vi for v in tuple_var for vi in self.collect_all_tuple_elements(self.collect_tuple_element(v))]
        else:
            return [tuple_var]

