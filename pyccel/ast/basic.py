#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
This module contains classes from which all pyccel nodes inherit.
They are:
- Basic, which provides a python AST
- TypedAstNode which describes each TypedAstNode
"""
import ast

from pyccel.utilities.stage   import PyccelStage

__all__ = ('Basic', 'Immutable', 'TypedAstNode', 'ScopedAstNode')

dict_keys   = type({}.keys())
dict_values = type({}.values())
iterable_types = (list, tuple, dict_keys, dict_values, set)
iterable = lambda x : isinstance(x, iterable_types)

pyccel_stage = PyccelStage()

#==============================================================================
class Immutable:
    """ Superclass for classes which cannot inherit
    from Basic """
    __slots__ = ()

#==============================================================================
class Basic:
    """
    Basic class from which all objects in the Pyccel AST inherit.

    This foundational class provides all the functionalities that are common to
    objects in the Pyccel AST. This includes the construction and navigation of
    the AST tree as well as an indication of the stage in which the object is
    valid (syntactic/semantic/etc).
    """
    __slots__ = ('_user_nodes', '_fst', '_recursion_in_progress' ,'_pyccel_staging')
    _ignored_types = (Immutable, type)
    _attribute_nodes = None

    def __init__(self):
        self._pyccel_staging = pyccel_stage.current_stage
        self._user_nodes = []
        self._fst = []
        self._recursion_in_progress = False
        for c_name in self._my_attribute_nodes:
            c = getattr(self, c_name)

            from pyccel.ast.literals import convert_to_literal

            if Basic._ignore(c):
                continue

            elif isinstance(c, (int, float, complex, str, bool)):
                # Convert basic types to literal types
                c = convert_to_literal(c)
                setattr(self, c_name, c)

            elif iterable(c):
                size = len(c)
                c = tuple(ci if (not isinstance(ci, (int, float, complex, str, bool)) \
                                 or Basic._ignore(ci)) \
                        else convert_to_literal(ci) for ci in c if not iterable(ci))
                if len(c) != size:
                    raise TypeError("Basic child cannot be a tuple of tuples")
                setattr(self, c_name, c)

            elif not isinstance(c, Basic):
                raise TypeError("Basic child must be a Basic or a tuple not {}".format(type(c)))


            if isinstance(c, tuple):
                for ci in c:
                    if not Basic._ignore(ci):
                        ci.set_current_user_node(self)
            else:
                c.set_current_user_node(self)

    @classmethod
    def _ignore(cls, c):
        """ Indicates if a node should be ignored when recursing
        """
        return c is None or isinstance(c, cls._ignored_types)

    def invalidate_node(self):
        """ Indicate that this node is temporary.
        This will allow it to remove itself from its attributes' users.
        If an attribute subsequently has no users, invalidate_node is called recursively
        """
        for c_name in self._my_attribute_nodes:
            c = getattr(self, c_name)

            if self._ignore(c):
                continue
            elif isinstance(c, tuple):
                _ = [ci.remove_user_node(self) for ci in c if not self._ignore(ci)]
            else:
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
        """ Returns all objects of the requested type
        in the current object

        Parameters
        ----------
        search_type : ClassType or tuple of ClassTypes
                      The types which we are looking for
        excluded_nodes : tuple of types
                      Types for which get_attribute_nodes should not be called

        Results
        -------
        list : List containing all objects of the
               requested type which exist in self
        """
        if self._recursion_in_progress:
            return []
        self._recursion_in_progress = True

        results = []
        for n in self._my_attribute_nodes:
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
        node : Basic
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
        node           : Basic
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

    def substitute(self, original, replacement, excluded_nodes = (), invalidate = True):
        """
        Substitute object 'original' for object 'replacement' in the code.
        Any types in excluded_nodes will not be visited

        Parameters
        ==========
        original    : object or tuple of objects
                      The original object to be replaced
        replacement : object or tuple of objects
                      The object which will be inserted instead
        excluded_nodes : tuple of types
                      Types for which substitute should not be called
        invalidate : bool
                    Indicates whether the removed object should
                    be invalidated
        """
        if self._recursion_in_progress:
            return
        self._recursion_in_progress = True

        if iterable(original):
            assert(iterable(replacement))
            assert(len(original) == len(replacement))
        else:
            original = (original,)
            replacement = (replacement,)

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

        for n in self._my_attribute_nodes:
            v = getattr(self, n)

            if isinstance(v, excluded_nodes):
                continue

            elif any(v is oi for oi in original):
                setattr(self, n, prepare_sub(v))

            elif isinstance(v, tuple):
                new_v = []
                for vi in v:
                    new_vi = vi
                    if not isinstance(vi, excluded_nodes):
                        if any(vi is oi for oi in original):
                            new_vi = prepare_sub(vi)
                        elif not self._ignore(vi):
                            vi.substitute(original, replacement, excluded_nodes, invalidate)
                    if iterable(new_vi):
                        new_v.extend(new_vi)
                    else:
                        new_v.append(new_vi)
                setattr(self, n, tuple(new_v))
            elif not self._ignore(v):
                v.substitute(original, replacement, excluded_nodes, invalidate)
        self._recursion_in_progress = False

    @property
    def is_atomic(self):
        """ Indicates whether the object has any attribute nodes.
        Returns true if it is an atom (no attribute nodes) and
        false otherwise
        """
        return not self._my_attribute_nodes

    def set_fst(self, fst):
        """Sets the python.ast fst."""
        if not isinstance(fst, ast.AST):
            raise TypeError("Fst must be an AST object, not {}".format(type(fst)))

        if self.fst:
            if hasattr(fst, 'lineno'):
                if self.fst.lineno != fst.lineno or self.fst.col_offset != fst.col_offset:
                    self._fst.append(fst)
        else:
            if not hasattr(fst, 'lineno'):
                # Handle module object
                fst.lineno     = 1
                fst.col_offset = 1

            self._fst.append(fst)

    @property
    def fst(self):
        if len(self._fst) == 1:
            return self._fst[0]
        else:
            return None

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
        This function should only be called in Basic
        """
        return self._user_nodes

    def get_direct_user_nodes(self, condition):
        """ For an object with multiple user nodes
        Get the objects which satisfy a given
        condition

        Parameters
        ----------
        condition : lambda
                    The condition which the user nodes
                    must satisfy to be returned
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
        """ Indicate that the current node is no longer used
        by the user_node. This function is usually called by
        the substitute method

        Parameters
        ----------
        user_node : Basic
                    Node which previously used the current node
        invalidate : bool
                    Indicates whether the removed object should
                    be invalidated
        """
        assert(user_node in self._user_nodes)
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

class TypedAstNode(Basic):
    """Class from which all nodes containing objects inherit
    """
    __slots__  = ()

    @property
    def shape(self):
        """ Tuple containing the length of each dimension
        of the object """
        return self._shape # pylint: disable=no-member

    @property
    def rank(self):
        """ Number of dimensions of the object
        """
        return self._rank # pylint: disable=no-member

    @property
    def dtype(self):
        """ Datatype of the object """
        return self._dtype # pylint: disable=no-member

    @property
    def precision(self):
        """ Precision of the datatype of the object """
        return self._precision # pylint: disable=no-member

    @property
    def order(self):
        """ Indicates whether the data is stored in
        row-major ('C') or column-major ('F') format.
        This is only relevant if rank > 1 """
        return self._order # pylint: disable=no-member

    def copy_attributes(self, x):
        self._shape     = x.shape
        self._rank      = x.rank
        self._dtype     = x.dtype
        self._precision = x.precision
        self._order     = x.order


#------------------------------------------------------------------------------
class ScopedAstNode(Basic):
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
