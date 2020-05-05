# coding: utf-8

from sympy import Symbol
from sympy.sets.sets import FiniteSet
from sympy.sets.sets import Union as sm_Union
from sympy.sets.sets import Intersection as sm_Intersection
from sympy.sets.sets import Complement
from sympy.sets.fancysets import Range as sm_Range
from sympy.sets.fancysets import Naturals

__all__ = (
    'Difference',
    'Group',
    'Intersection',
    'Range',
    'Split',
    'Union',
    'UniversalGroup'
)

#==============================================================================
class Group(FiniteSet):
    """Represents a group of processes.

    processes: Set
        set of the processes within the group

    rank: int
        the rank of the calling process within the group

    Examples

    >>> from pyccel.ast.parallel.group import Group
    >>> g = Group(1, 2, 3, 4)
    >>> g
    {1, 2, 3, 4}
    >>> g.size
    4
    """
    _comm = None

    @property
    def size(self):
        """the number of processes in the group."""
        return len(self)

    @property
    def processes(self):
        """Returns the set of the group processes."""
        return self._args[0]

    @property
    def rank(self):
        return self._args[1]

    @property
    def communicator(self):
        return self._comm

    def set_communicator(self, comm):
        self._comm = comm

    def translate(self, rank1, other, rank2):
        """
        This routine takes an array of n ranks (ranks1) which
        are ranks of processes in self. It returns in
        ranks2 the corresponding ranks of the processes
        as they are in other.
        MPI_UNDEFINED is returned for processes not in
        other
        """
        pass

    def compare(self, other):
        """
        This routine returns the relationship between self
        and other
        If self and other contain the same processes,
        ranked the same way, this routine returns
        MPI_IDENT
        If self and other contain the same processes,
        but ranked differently, this routine returns
        MPI_SIMILAR
        Otherwise this routine returns MPI_UNEQUAL.
        """
        pass


    def union(self, other):
        """
        Returns in newgroup a group consisting
        of all processes in self followed by all
        processes in other, with no duplication.

        Examples

        As a shortcut it is possible to use the '+' operator:

        >>> from pyccel.ast.parallel.group import Group
        >>> g1 = Group(1, 3, 6, 7, 8, 9)
        >>> g2 = Group(0, 2, 4, 5, 8, 9)
        >>> g1.union(g2)
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        >>> g1 + g2
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        """
        return Union(self, other)

    def intersection(self, other):
        """
        Returns in newgroup all processes
        that are in both groups, ordered as in self.

        Examples

        >>> from pyccel.ast.parallel.group import Group
        >>> g1 = Group(1, 3, 6, 7, 8, 9)
        >>> g2 = Group(0, 2, 4, 5, 8, 9)
        >>> g1.intersection(g2)
        {8, 9}
        """
        return Intersection(self, other)

    def difference(self, other):
        """
        Returns in newgroup all processes
        in self that are not in other, ordered as
        in self.

        Examples

        As a shortcut it is possible to use the '-' operator:

        >>> from pyccel.ast.parallel.group import Group
        >>> g1 = Group(1, 3, 6, 7, 8, 9)
        >>> g2 = Group(0, 2, 4, 5, 8, 9)
        >>> g1.difference(g2)
        {1, 3, 6, 7}
        >>> g1 - g2
        {1, 3, 6, 7}
        """
        return Difference(self, other)

#==============================================================================
class Union(sm_Union):
    """
    Represents the union of groups.

    Examples

    >>> from pyccel.ast.parallel.group import Group, Union
    >>> g1 = Group(1, 3, 6, 7, 8, 9)
    >>> g2 = Group(0, 2, 4, 5, 8, 9)
    >>> Union(g1, g2)
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    """

    def __new__(cls, *args, **kwargs):
        u = sm_Union(*args, **kwargs)

        return Group(*u.args)

#==============================================================================
class Intersection(sm_Intersection):
    """
    Represents the intersection of groups.

    Examples

    >>> from pyccel.ast.parallel.group import Group, Intersection
    >>> g1 = Group(1, 3, 6, 7, 8, 9)
    >>> g2 = Group(0, 2, 4, 5, 8, 9)
    >>> Intersection(g1, g2)
    {8, 9}
    """

    def __new__(cls, *args, **kwargs):
        i = sm_Intersection(*args, **kwargs)

        return Group(*i.args)

#==============================================================================
class Difference(Complement):
    """
    Represents the difference between two groups.

    Examples

    >>> from pyccel.ast.parallel.group import Group, Difference
    >>> g1 = Group(1, 3, 6, 7, 8, 9)
    >>> g2 = Group(0, 2, 4, 5, 8, 9)
    >>> Difference(g1, g2)
    {1, 3, 6, 7}
    """

    def __new__(cls, *args, **kwargs):
        c = Complement(*args, **kwargs)

        return Group(*c.args)

#==============================================================================
class Range(sm_Range):
    """
    Representes a range of processes.

    Examples

    >>> from pyccel.ast.parallel.group import Range
    >>> from sympy import Symbol
    >>> n = Symbol('n')
    >>> Range(0, n)
    Range(0, n)
    """

    def __new__(cls, *args):
        _args = []
        for a in args:
            if isinstance(a, Symbol):
                _args.append(0)
            else:
                _args.append(a)
        r = sm_Range.__new__(cls, *_args)
        r._args = args

        return r

#==============================================================================
class UniversalGroup(Naturals):
    """
    Represents the group of all processes.
    Since the number of processes is only known at the run-time and the universal
    group is assumed to contain all processes, it is convinient to consider it
    as the set of all possible processes which is nothing else than the set of
    Natural numbers.

    np: Symbol
        a sympy symbol describing the total number of processes.
    """

    np = Symbol('np')

    @property
    def processes(self):
        return Range(0, self.np)

    @property
    def communicator(self):
        from pyccel.ast.parallel.communicator import UniversalCommunicator
        return UniversalCommunicator()

    @property
    def size(self):
        """the total number of processes."""
        return self.np

#==============================================================================
# TODO check size between colors and group
class Split(Group):
    """
    Splits the group over a given color.

    Examples

    >>> from pyccel.ast.parallel.group import UniversalGroup
    >>> from pyccel.ast.parallel.communicator import UniversalCommunicator, split
    >>> colors = [1, 1, 0, 1, 0, 0]
    >>> g = Split(UniversalGroup(), colors, 0)
    >>> g
    {2, 4, 5}
    >>> comm = split(UniversalCommunicator(), g)
    Communicator({2, 4, 5}, None)
    """

    def __new__(cls, group, colors, color):
        args = []
        i = 0
        while (i < len(colors)):
            if colors[i] == color:
                args.append(i)
            i += 1

        return Group.__new__(cls, *args)
