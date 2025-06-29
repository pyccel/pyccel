#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#

"""
This module contains classes from which all pyccel nodes inherit. They are:

- PyccelAstNode, which provides a base class for our Python AST nodes;
- TypedAstNode, which inherits from PyccelAstNode and provides a base class for
  AST nodes requiring type descriptors.
"""
import ast
from types import GeneratorType

from pyccel.utilities.stage   import PyccelStage

__all__ = ('Immutable', 'PyccelAstNode', 'ScopedAstNode', 'TypedAstNode')

dict_keys   = type({}.keys())
dict_values = type({}.values())
def iterable(x):
    """
    Determine if type is iterable for a PyccelAstNode.

    Determine if type is iterable for a PyccelAstNode. This looks for iterable
    values but excludes arbitrary types which implement `__iter__` to avoid
    iterating over unexpected types (e.g Variable).

    Parameters
    ----------
    x : object
        Any Python object to be examined.

    Returns
    -------
    bool
        True if object is iterable for a PyccelAstNode.
    """
    return isinstance(x, (list, tuple, dict_keys, dict_values, set, GeneratorType))

pyccel_stage = PyccelStage()

#==============================================================================
class Immutable:
    """ Superclass for classes which cannot inherit
    from PyccelAstNode """
    __slots__ = ()

#==============================================================================
class PyccelAstNode:
    """
    PyccelAstNode class from which all objects in the Pyccel AST inherit.

    This foundational class provides all the functionalities that are common to
    objects in the Pyccel AST. This includes the construction and navigation of
    the AST tree as well as an indication of the stage in which the object is
    valid (syntactic/semantic/etc).
    """
    __slots__ = ('_user_nodes', '_ast', '_recursion_in_progress' ,'_pyccel_staging')
    _ignored_types = (Immutable, type)
    _attribute_nodes = None

    def __init__(self):
        self._pyccel_staging = pyccel_stage.current_stage
        self._user_nodes = []
        self._ast = []
        self._recursion_in_progress = False
        for c_name in self._my_attribute_nodes: #pylint: disable=not-an-iterable
            c = getattr(self, c_name)

            from pyccel.ast.literals import convert_to_literal

            if PyccelAstNode._ignore(c):
                continue

            elif isinstance(c, (int, float, complex, str, bool)):
                # Convert basic types to literal types
                c = convert_to_literal(c)
                setattr(self, c_name, c)

            elif iterable(c):
                size = len(c)
                c = tuple(ci if (not isinstance(ci, (int, float, complex, str, bool)) \
                                 or PyccelAstNode._ignore(ci)) \
                        else convert_to_literal(ci) for ci in c if not iterable(ci))
                if len(c) != size:
                    raise TypeError("PyccelAstNode child cannot be a tuple of tuples")
                setattr(self, c_name, c)

            elif not isinstance(c, PyccelAstNode):
                raise TypeError(f"PyccelAstNode child must be a Basic or a tuple not {type(c)}")


            if isinstance(c, tuple):
                for ci in c:
                    if not PyccelAstNode._ignore(ci):
                        ci.set_current_user_node(self)
            else:
                c.set_current_user_node(self)

    @classmethod
    def _ignore(cls, c):
        """ Indicates if a node should be ignored when recursing
        """
        return c is None or isinstance(c, cls._ignored_types)

    def invalidate_node(self):
        """
        Indicate that this node is no longer used.

        Indicate that this node is temporary and is no longer used.
        This will allow it to remove itself from its attributes' users.
        If an attribute subsequently has no users, invalidate_node is called recursively.
        This prevents the tree from becoming filled with temporary objects and prevents
        obsolete objects being retrieved when searching for attribute nodes.
        """
        for c_name in self._my_attribute_nodes: #pylint: disable=not-an-iterable
            c = getattr(self, c_name)

            if self._ignore(c):
                continue
            elif isinstance(c, tuple):
                _ = [ci.remove_user_node(self) for ci in c if not self._ignore(ci) \
                        and ci.pyccel_staging == self.pyccel_staging]
            elif c.pyccel_staging == self.pyccel_staging:
                # Pyccel stage can change for basic objects with no attributes (e.g. Literal, Pass, etc)
                c.remove_user_node(self)

    def get_user_nodes(self, search_type, excluded_nodes = ()):
        """ Returns all objects of the requested type
        which use the current object

        Parameters
        ----------
        search_type : ClassType or tuple of ClassTypes
                      The types which we are looking for
        excluded_nodes : tuple of types
                      Types for which get_user_nodes should not be called

        Results
        -------
        list : List containing all objects of the
               requested type which contain self
        """
        if self._recursion_in_progress or len(self._user_nodes) == 0:
            return []
        else:
            self._recursion_in_progress = True

            results  = [p for p in self._user_nodes if     isinstance(p, search_type) and \
                                                       not isinstance(p, excluded_nodes)]

            results += [r for p in self._user_nodes if not self._ignore(p) and \
                                                       not isinstance(p, (search_type, excluded_nodes)) \
                          for r in p.get_user_nodes(search_type, excluded_nodes = excluded_nodes)]
            self._recursion_in_progress = False
            return results

    def get_attribute_nodes(self, search_type, excluded_nodes = ()):
        """
        Get all objects of the requested type in the current object.

        Returns all objects of the requested type which are stored in the
        current object.

        Parameters
        ----------
        search_type : ClassType or tuple of ClassTypes
                      The types which we are looking for.
        excluded_nodes : tuple of types
                      Types for which get_attribute_nodes should not be called.

        Returns
        -------
        list
            List containing all objects of the requested type which exist in self.
        """
        if self._recursion_in_progress:
            return []
        self._recursion_in_progress = True

        results = []
        for n in self._my_attribute_nodes: #pylint: disable=not-an-iterable
            v = getattr(self, n)

            if isinstance(v, excluded_nodes):
                continue

            elif isinstance(v, search_type):
                results.append(v)

            elif isinstance(v, tuple):
                for vi in v:
                    if isinstance(vi, excluded_nodes):
                        continue
                    elif isinstance(vi, search_type):
                        results.append(vi)
                    elif not self._ignore(vi):
                        results.extend(vi.get_attribute_nodes(
                            search_type, excluded_nodes=excluded_nodes))

            elif not self._ignore(v):
                results.extend(v.get_attribute_nodes(
                    search_type, excluded_nodes = excluded_nodes))

        self._recursion_in_progress = False
        return results

    def is_attribute_of(self, node):
        """ Identifies whether this object is an attribute of node.
        The function searches recursively down the attribute tree.

        Parameters
        ----------
        node : PyccelAstNode
               The object whose attributes we are interested in

        Results
        -------
        bool
        """
        return node.is_user_of(self)

    def is_user_of(self, node, excluded_nodes = ()):
        """ Identifies whether this object is a user of node.
        The function searches recursively up the user tree

        Parameters
        ----------
        node           : PyccelAstNode
                      The object whose users we are interested in
        excluded_nodes : tuple of types
                      Types for which is_user_of should not be called

        Results
        -------
        bool
        """
        if node.recursion_in_progress:
            return []
        node.toggle_recursion()

        for v in node.get_all_user_nodes():

            if v is self:
                node.toggle_recursion()
                return True

            elif isinstance(v, excluded_nodes):
                continue

            elif not self._ignore(v):
                res = self.is_user_of(v, excluded_nodes=excluded_nodes)
                if res:
                    node.toggle_recursion()
                    return True

        node.toggle_recursion()
        return False

    def substitute(self, original, replacement, excluded_nodes = (), invalidate = True,
            is_equivalent = None):
        """
        Substitute object 'original' for object 'replacement' in the code.

        Substitute object 'original' for object 'replacement' in the code.
        Any types in excluded_nodes will not be visited.

        Parameters
        ----------
        original : object or tuple of objects
                      The original object to be replaced.
        replacement : object or tuple of objects
                      The object which will be inserted instead.
        excluded_nodes : tuple of types
                      Types for which substitute should not be called.
        invalidate : bool
                    Indicates whether the removed object should
                    be invalidated.
        is_equivalent : function, optional
                    A function that compares the original object to the object
                    in the PyccelAstNode to determine if it is the object that
                    we are searching for. Usually this is an equality in the
                    syntactic stage and an identity comparison in the semantic
                    stage, but occasionally a different choice may be useful.
        """
        if self._recursion_in_progress:
            return
        self._recursion_in_progress = True

        if not original:
            assert not replacement
            self._recursion_in_progress = False
            return

        if iterable(original):
            assert iterable(replacement)
            assert len(original) == len(replacement)
        else:
            original = (original,)
            replacement = (replacement,)

        if is_equivalent is None:
            if self.pyccel_staging == 'syntactic':
                is_equivalent = lambda x, y: x == y #pylint:disable=unnecessary-lambda-assignment
            else:
                is_equivalent = lambda x, y: x is y #pylint:disable=unnecessary-lambda-assignment

        def prepare_sub(found_node):
            idx = original.index(found_node)
            rep = replacement[idx]
            if iterable(rep):
                for r in rep:
                    if not self._ignore(r):
                        r.set_current_user_node(self)
            else:
                if not self._ignore(rep):
                    rep.set_current_user_node(self)
            if not self._ignore(found_node):
                found_node.remove_user_node(self, invalidate)
            return rep

        for n in self._my_attribute_nodes: #pylint: disable=not-an-iterable
            v = getattr(self, n)

            if isinstance(v, excluded_nodes):
                continue

            elif any(is_equivalent(v, oi) for oi in original):
                setattr(self, n, prepare_sub(v))

            elif isinstance(v, tuple):
                new_v = []
                for vi in v:
                    new_vi = vi
                    if not isinstance(vi, excluded_nodes):
                        if any(is_equivalent(vi, oi) for oi in original):
                            new_vi = prepare_sub(vi)
                        elif not self._ignore(vi):
                            vi.substitute(original, replacement, excluded_nodes, invalidate, is_equivalent)
                    if iterable(new_vi):
                        new_v.extend(new_vi)
                    else:
                        new_v.append(new_vi)
                setattr(self, n, tuple(new_v))
            elif not self._ignore(v):
                v.substitute(original, replacement, excluded_nodes, invalidate, is_equivalent)
        self._recursion_in_progress = False

    @property
    def is_atomic(self):
        """ Indicates whether the object has any attribute nodes.
        Returns true if it is an atom (no attribute nodes) and
        false otherwise
        """
        return not self._my_attribute_nodes

    @property
    def python_ast(self):
        """
        Get an `ast.AST` object describing the parsed code that this node represents.

        Get the AST (abstract syntax tree) object which Python parsed
        in the original code. This object describes the Python code being
        translated. It provides line numbers and columns which can be
        used to report the origin of any potential errors.
        If this object appears in multiple places in the code (e.g. Variables) then
        this property returns `None` so as not to accidentally print the wrong
        location.

        Returns
        -------
        ast.AST
            The AST object which was parsed.
        """
        if len(self._ast) == 1:
            return self._ast[0]
        else:
            return None

    def set_current_ast(self, ast_node):
        """
        Set the `ast.AST` object which describes the parsed code that this node currently represents.

        Set the AST (abstract syntax tree) object which Python parsed in the original code and which
        resulted in the creation (or use) of this PyccelAstNode. This object describes the Python code
        being translated. It provides line numbers and columns which can be used to report the origin
        of any potential errors. If this function is called multiple times then accessing the AST
        object will result in `None` so as not to accidentally print the wrong code location.

        Parameters
        ----------
        ast_node : ast.AST
            The AST object which was parsed.
        """
        if not isinstance(ast_node, ast.AST):
            raise TypeError(f"ast_node must be an AST object, not {type(ast_node)}")

        if self.python_ast:
            if hasattr(ast_node, 'lineno'):
                if self.python_ast.lineno != ast_node.lineno or self.python_ast.col_offset != ast_node.col_offset:
                    self._ast.append(ast_node)
        else:
            if not hasattr(ast_node, 'lineno'):
                # Handle module object
                ast_node.lineno     = 1
                ast_node.col_offset = 1

            self._ast.append(ast_node)


    def toggle_recursion(self):
        """ Change the recursion state
        """
        self._recursion_in_progress = not self._recursion_in_progress

    @property
    def recursion_in_progress(self):
        """ Recursion state used to avoid infinite loops
        """
        return self._recursion_in_progress

    def get_all_user_nodes(self):
        """ Returns all the objects user nodes.
        This function should only be called in PyccelAstNode
        """
        return self._user_nodes

    def get_direct_user_nodes(self, condition):
        """
        Get the direct user nodes which satisfy the condition.

        This function returns all the direct user nodes which satisfy the
        provided condition. A "direct" user node is a node which uses the
        instance directly (e.g. a `FunctionCall` uses a `FunctionDef` directly
        while a `FunctionDef` uses a `Variable` indirectly via a `FunctionDefArgument`
        or a `CodeBlock`). Most objects only have 1 direct user node so
        this function only makes sense for an object with multiple user nodes.
        E.g. a `Variable`, or a `FunctionDef`.

        Parameters
        ----------
        condition : lambda
            The condition which the user nodes must satisfy to be returned.

        Returns
        -------
        list
            The user nodes which satisfy the condition.
        """
        return [p for p in self._user_nodes if condition(p)]

    def set_current_user_node(self, user_nodes):
        """ Inform the class about the most recent user of the node
        """
        self._user_nodes.append(user_nodes)

    @property
    def current_user_node(self):
        """ Get the user node for an object with only one user node
        """
        assert len(self._user_nodes) == 1
        return self._user_nodes[0]

    def clear_syntactic_user_nodes(self):
        """
        Delete all information about syntactic user nodes.

        Delete all user nodes which are only valid for the syntactic
        stage from the list of user nodes. This is useful
        if the same node is used for the syntactic and semantic
        stages.
        """
        self._user_nodes = [u for u in self._user_nodes if u.pyccel_staging != 'syntactic']

    def remove_user_node(self, user_node, invalidate = True):
        """
        Remove the specified user node from the AST tree.

        Indicate that the current node is no longer used by the user_node.
        This function is usually called by the substitute method. It removes
        the specified user node from the user nodes internal property
        meaning that the node cannot appear in the results when searching
        through the tree.

        Parameters
        ----------
        user_node : PyccelAstNode
            Node which previously used the current node.
        invalidate : bool
            Indicates whether the removed object should be invalidated.
        """
        assert user_node in self._user_nodes
        self._user_nodes.remove(user_node)
        if self.is_unused and invalidate:
            self.invalidate_node()

    @property
    def is_unused(self):
        """ Indicates whether the class has any users
        """
        return len(self._user_nodes)==0

    @property
    def _my_attribute_nodes(self):
        """ Getter for _attribute_nodes to avoid codacy warnings
        about no-member. This attribute must be instantiated in
        the subclasses and this ensures that an error is raised
        if it isn't
        """
        return self._attribute_nodes # pylint: disable=no-member

    @property
    def pyccel_staging(self):
        """
        Indicate the stage at which the object was created.

        Indicate the stage at which the object was created [syntactic/semantic/codegen/cwrapper].
        """
        return self._pyccel_staging

    def update_pyccel_staging(self):
        """
        Indicate that an object has been updated and is now valid in the current pyccel stage.

        Indicate that an object has been updated and is now valid in the current pyccel stage.
        This results in the pyccel_staging being updated to match the current stage.
        """
        self._pyccel_staging = pyccel_stage.current_stage

class TypedAstNode(PyccelAstNode):
    """
    Class from which all typed objects inherit.

    The class from which all objects which can be described with type information
    must inherit. Objects with type information are objects which take up memory
    in a running program (e.g. a variable or the result of a function call).
    Each typed object is described by an underlying datatype, a rank,
    a shape, and a data layout ordering.
    """
    __slots__  = ()

    @property
    def shape(self):
        """
        Tuple containing the length of each dimension of the object or None.

        A tuple containing the length of each dimension of the object if the object
        is an array (with rank>0). Otherwise None.
        """
        return self._shape # pylint: disable=no-member

    @property
    def rank(self):
        """
        Number of dimensions of the object.

        Number of dimensions of the object. If the object is a scalar then
        this is equal to 0.
        """
        return self.class_type.rank

    @property
    def dtype(self):
        """
        Datatype of the object.

        The underlying datatype of the object. In the case of scalars this is
        equivalent to the type of the object in Python. For objects in (homogeneous)
        containers (e.g. list/ndarray/tuple), this is the type of an arbitrary element
        of the container.
        """
        return self.class_type.datatype

    @property
    def order(self):
        """
        The data layout ordering in memory.

        Indicates whether the data is stored in row-major ('C') or column-major
        ('F') format. This is only relevant if rank > 1. When it is not relevant
        this function returns None.
        """
        return self.class_type.order

    @property
    def class_type(self):
        """
        The type of the object.

        The Python type of the object. In the case of scalars this is equivalent to
        the datatype. For objects in (homogeneous) containers (e.g. list/ndarray/tuple),
        this is the type of the container.
        """
        return self._class_type # pylint: disable=no-member

    @classmethod
    def static_type(cls):
        """
        The type of the object.

        The Python type of the object. In the case of scalars this is equivalent to
        the datatype. For objects in (homogeneous) containers (e.g. list/ndarray/tuple),
        this is the type of the container.

        This function is static and will return an AttributeError if the
        class does not have a predetermined order.
        """
        return cls._static_type # pylint: disable=no-member

    def copy_attributes(self, x):
        """
        Copy the attributes describing a TypedAstNode into this node.

        Copy the attributes which describe the TypedAstNode passed as
        argument (dtype, shape, rank, order) into this node
        so that the two nodes can be stored in the same object.

        Parameters
        ----------
        x : TypedAstNode
            The node from which the attributes should be copied.
        """
        self._shape      = x.shape
        self._class_type = x.class_type


#------------------------------------------------------------------------------
class ScopedAstNode(PyccelAstNode):
    """ Class from which all objects with a scope inherit
    """
    __slots__ = ('_scope',)

    def __init__(self, scope = None):
        self._scope = scope
        super().__init__()

    @property
    def scope(self):
        """ Local scope of the current object
        This contains all available objects in this part of the code
        """
        return self._scope
