#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
This module contains classes from which all pyccel nodes inherit.
They are:
- Basic, which provides a python AST
- PyccelAstNode which describes each PyccelAstNode
"""
import ast
from sympy.core.basic import Basic as sp_Basic

__all__ = ('Basic', 'PyccelAstNode')

dict_keys   = {}.keys().__class__
dict_values = {}.values().__class__
iterable_types = (list, tuple, dict_keys, dict_values)

#==============================================================================
class Basic(sp_Basic):
    """Basic class for Pyccel AST."""
    _fst = None

    def __new__(cls, *args, **kwargs):
        hashable_args  = [a if not isinstance(a, list) else tuple(a) for a in args]
        return sp_Basic.__new__(cls, *hashable_args)

    def __init__(self):
        self._user_nodes = []
        self._fst = []
        for c_name in self._attribute_nodes:
            c = getattr(self, c_name)
            from pyccel.ast.internals import PyccelSymbol
            from pyccel.ast.literals import convert_to_literal
            if isinstance(c, PyccelSymbol):
                # Anti-pattern
                # PyccelSymbol is not a Basic so it must be handled separately
                continue
            elif isinstance(c, (int, float, complex, str, bool)):
                # Convert basic types to literal types
                c = convert_to_literal(c)
                setattr(self, c_name, c)

            elif isinstance(c, iterable_types):
                if any(isinstance(ci, iterable_types) for ci in c):
                    raise TypeError("Basic child cannot be a tuple of tuples")
                c = tuple(ci if (not isinstance(ci, (int, float, complex, str, bool)) \
                                 or isinstance(ci, PyccelSymbol)) \
                        else convert_to_literal(ci) for ci in c)
                setattr(self, c_name, c)

            elif not isinstance(c, Basic) and c is not None:
                raise TypeError("Basic child must be a Basic or a tuple not {}".format(type(c)))

            if isinstance(c, tuple):
                for ci in c:
                    if ci and not isinstance(ci, PyccelSymbol):
                        ci.set_current_user_node(self)
            elif c and not isinstance(c, PyccelSymbol):
                c.set_current_user_node(self)

    def invalidate_node(self):
        """ Indicate that this node is temporary.
        This will allow it to remove itself from its children's users.
        If a child subsequently has no users, invalidate_node is called recursively
        """
        for c_name in self._attribute_nodes:
            c = getattr(self, c_name)
            from pyccel.ast.internals import PyccelSymbol
            from pyccel.ast.literals import convert_to_literal
            if isinstance(c, PyccelSymbol):
                # Anti-pattern
                # PyccelSymbol is not a Basic so it must be handled separately
                continue

            if isinstance(c, tuple):
                for ci in c:
                    if ci and not isinstance(ci, PyccelSymbol):
                        ci.remove_user_node(self)
                        if not ci._user_nodes:
                            ci.invalidate_node()
            elif c and not isinstance(c, PyccelSymbol):
                c.remove_user_node(self)
                if not c._user_nodes:
                    c.invalidate_node()

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
        if len(self._user_nodes) == 0:
            return []
        else:
            from pyccel.ast.internals import PyccelSymbol

            results  = [p for p in self._user_nodes if isinstance(p, search_type) and not isinstance(p, excluded_nodes)]
            results += [r for p in self._user_nodes if not isinstance(p, (search_type, PyccelSymbol, excluded_nodes)) \
                    for r in p.get_user_nodes(search_type, excluded_nodes = excluded_nodes)]
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
        from pyccel.ast.internals import PyccelSymbol

        results = []
        for n in self._attribute_nodes:
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

                    elif vi is not None and not isinstance(vi, PyccelSymbol):
                        results.extend(vi.get_attribute_nodes(
                            search_type, excluded_nodes=excluded_nodes))

            elif v is not None and not isinstance(v, PyccelSymbol):
                results.extend(v.get_attribute_nodes(
                    search_type, excluded_nodes = excluded_nodes))

        return results

    def substitute(self, original, replacement, excluded_nodes = ()):
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
        """
        from pyccel.ast.internals import PyccelSymbol

        if isinstance(original, tuple):
            assert(isinstance(replacement, tuple))
            assert(len(original) == len(replacement))
        else:
            original = (original,)
            replacement = (replacement,)

        for n in self._attribute_nodes:
            v = getattr(self, n)
            if isinstance(v, excluded_nodes):
                continue
            elif v in original:
                idx = original.index(v)
                v.remove_user_node(self)
                setattr(self, n, replacement[idx])
                replacement[idx].set_current_user_node(self)
            elif isinstance(v, tuple):
                new_v = []
                for vi in v:
                    new_vi = vi
                    if vi in original:
                        idx = original.index(vi)
                        vi.remove_user_node(self)
                        new_vi = replacement[idx]
                        replacement[idx].set_current_user_node(self)
                    elif not isinstance(vi, excluded_nodes):
                        vi.substitute(original, replacement, excluded_nodes)
                    new_v.append(new_vi)
                setattr(self, n, tuple(new_v))
            elif v is not None and not isinstance(v, PyccelSymbol):
                v.substitute(original, replacement, excluded_nodes)

    @property
    def is_atomic(self):
        """ Indicates whether the object has any attribute nodes
        """
        return bool(self._attribute_nodes)

    def set_fst(self, fst):
        """Sets the python.ast fst."""
        if not isinstance(fst, ast.AST):
            raise TypeError("Fst must be an AST object, not {}".format(type(fst)))

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
        self._user_nodes.append(user_nodes)

    def remove_user_node(self, user_node):
        """ Indicate that the current node is no longer used
        by the user_node. This function is usually called by
        the substitute method

        Parameters
        ----------
        user_node : Basic
                    Node which previously used the current node
        """
        assert(user_node in self._user_nodes)
        self._user_nodes.remove(user_node)

    def __eq__(self, other):
        #TODO: Remove with sympy inheritance
        return id(self) == id(other)

    def __hash__(self):
        #TODO: Remove with sympy inheritance
        return id(self)

class PyccelAstNode(Basic):
    """Class from which all nodes containing objects inherit
    """
    stage      = None
    _shape     = None
    _rank      = None
    _dtype     = None
    _precision = None
    _order     = None

    @property
    def shape(self):
        """ Tuple containing the length of each dimension
        of the object """
        return self._shape

    @property
    def rank(self):
        """ Number of dimensions of the object
        """
        return self._rank

    @property
    def dtype(self):
        """ Datatype of the object """
        return self._dtype

    @property
    def precision(self):
        """ Precision of the datatype of the object """
        return self._precision

    @property
    def order(self):
        """ Indicates whether the data is stored in
        row-major ('C') or column-major ('F') format.
        This is only relevant if rank > 1 """
        return self._order

    def copy_attributes(self, x):
        self._shape     = x.shape
        self._rank      = x.rank
        self._dtype     = x.dtype
        self._precision = x.precision
        self._order     = x.order

