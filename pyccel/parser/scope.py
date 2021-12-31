from collections import OrderedDict


class Scope(object):
    """ Class representing all objects defined within a given
    scope

    Parameters
    ----------
    decorators : OrderedDict
                 A dictionary of any decorators which operate on
                 objects in this scope
    """

    def __init__(self, *, decorators=None):

        keys = ('functions','variables','classes',
                'imports','python_functions','symbolic_functions',
                'macros','templates','headers','decorators',
                'static_functions','cls_constructs')

        self._imports = OrderedDict((k,OrderedDict()) for k in keys)

        self._locals  = OrderedDict((k,OrderedDict()) for k in keys)

        if decorators:
            self._locals['decorators'].update(decorators)

        # TODO use another name for headers
        #      => reserved keyword, or use __
        self.parent_scope        = None
        self._sons_scopes        = OrderedDict()

        self._used_symbols = {}

        self._is_loop = False
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

        child = Scope(**kwargs)

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
    def static_functions(self):
        """ A dictionary of static functions defined in this scope
        """
        return self._locals['static_functions']

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
        """ 
        """
        return self._locals['symbolic_functions']

    @property
    def python_functions(self):
        """ 
        """
        return self._locals['python_functions']

    def find_in_scope(self, name, location):
        if name in self._locals[location]:
            return self._locals[location][name]

        if name in self.imports[location]:
            return self.imports[location][name]

        # Walk up the tree of Scope objects, until the root if needed
        if self.parent_scope:
            return self.parent_scope.find_variable(name)
        else:
            raise RuntimeError("Variable not found in scope")

    @property
    def is_loop(self):
        return self._is_loop

    @property
    def loops(self):
        return self._loops

