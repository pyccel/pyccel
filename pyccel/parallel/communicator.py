# coding: utf-8

from sympy.core.basic import Basic
from sympy.sets.sets import Set


class UniversalCommunicator(Basic):
    """Represents the communicator to all processes."""

    @property
    def group(self):
        from pyccel.parallel.group import UniversalGroup
        return UniversalGroup()

class Communicator(Basic):
    """Represents a communicator in the code.

    size: int
        the number of processes in the group associated with the
        communicator

    rank: int
        the rank of the calling process within the group associated
        with the communicator

    Examples

    >>> from pyccel.ast.mpi import Communicator
    >>> Communicator()
    mpi_comm_world
    """

    def __new__(cls, group, rank=None):
        from pyccel.parallel.group import UniversalGroup
        comm = Basic.__new__(cls, group, rank)
        if not isinstance(group, UniversalGroup):
            group.set_communicator(comm)
        return comm

    @property
    def rank(self):
        return self._args[0]

    @property
    def group(self):
        """the group associated with the communicator."""
        return self._args[1]

    @property
    def size(self):
        return self.group.size

    def duplicate(self):
        """This routine creates a duplicate of the communicator other
        has the same fixed attributes as the communicator.

        Examples

        >>> from pyccel.ast.mpi import Communicator
        >>> c = Communicator(is_root=True)
        >>> o = c.duplicate()
        """
        rank    = self.rank
        group   = self.group
        return Communicator(rank=rank, group=group)

    def split(self, group):
        """Split the communicator over colors by returning the new comm
        associated to the processes with defined color.

        colors: iterable
            map over all available processes.
        """
        pass


def split(comm, group, rank=None):
    """Splits the communicator over a given color."""
    from pyccel.parallel.group import Split
    if not isinstance(group, Split):
        raise TypeError("Expecting a group of instance Split.")

    c = Communicator(group, rank)
    group.set_communicator(c)
    return c
