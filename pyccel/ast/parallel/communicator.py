# coding: utf-8

from pyccel.ast.parallel.basic import Basic
from pyccel.ast.parallel.group import UniversalGroup
from pyccel.ast.parallel.group import Split

__all__ = (
    'Communicator',
    'UniversalCommunicator',
    'split'
)

#==============================================================================
class UniversalCommunicator(Basic):
    """Represents the communicator to all processes."""

    @property
    def size(self):
        return self.group.size

    @property
    def group(self):
        return UniversalGroup()

#==============================================================================
class Communicator(Basic):
    """
    Represents a communicator in the code.

    size: int
        the number of processes in the group associated with the
        communicator

    rank: int
        the rank of the calling process within the group associated
        with the communicator

    Examples

    """

    def __new__(cls, group, rank=None):
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

        """
        rank    = self.rank
        group   = self.group
        return Communicator(rank=rank, group=group)

    def split(self, group):
        """
        Split the communicator over colors by returning the new comm
        associated to the processes with defined color.

        Examples

        """
        pass

#==============================================================================
def split(comm, group, rank=None):
    """
    Splits the communicator over a given color.

    Examples

    """
    if not isinstance(group, Split):
        raise TypeError("Expecting a group of instance Split.")

    c = Communicator(group, rank)
    group.set_communicator(c)
    return c
