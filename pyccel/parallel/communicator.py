# coding: utf-8

from sympy.core.basic import Basic as sm_Basic
from sympy.sets.sets import Set

class Basic(sm_Basic):
    is_integer = False
    _dtypes = {}
    _dtypes['size'] = 'int'
    _dtypes['rank'] = 'int'

    def __new__(cls, *args, **options):
        return super(Basic, cls).__new__(cls, *args, **options)

    def dtype(self, attr):
        """Returns the datatype of a given attribut/member."""
        return self._dtypes[attr]

class UniversalCommunicator(Basic):
    """Represents the communicator to all processes."""

    @property
    def size(self):
        return self.group.size

    @property
    def group(self):
        from pyccel.parallel.group import UniversalGroup
        return UniversalGroup()

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

def Size(Basic):
    is_integer = True

    def __new__(cls, *args, **kwargs):
        return super(Size, cls).__new__(*args, **kwargs)

    @property
    def communicator(self):
        self._args[0]

def split(comm, group, rank=None):
    """
    Splits the communicator over a given color.

    Examples

    """
    from pyccel.parallel.group import Split
    if not isinstance(group, Split):
        raise TypeError("Expecting a group of instance Split.")

    c = Communicator(group, rank)
    group.set_communicator(c)
    return c
