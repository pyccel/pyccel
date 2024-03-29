# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
OpenMP has several constructs and directives, and this file contains the OpenMP types that are supported.
We represent some types with the OmpAnnotatedComment type.
These types are detailed on our documentation:
https://github.com/pyccel/pyccel/blob/master/tutorial/openmp.md
"""

from .basic import PyccelAstNode

__all__ = ('OmpAnnotatedComment',
           'OMP_For_Loop',
           'OMP_Simd_Construct',
           'OMP_TaskLoop_Construct',
           'OMP_Distribute_Construct',
           'OMP_Parallel_Construct',
           'OMP_Task_Construct',
           'OMP_Single_Construct',
           'OMP_Critical_Construct',
           'OMP_Master_Construct',
           'OMP_Masked_Construct',
           'OMP_Cancel_Construct',
           'OMP_Target_Construct',
           'OMP_Teams_Construct',
           'OMP_Sections_Construct',
           'OMP_Section_Construct',
           'Omp_End_Clause')

class OmpAnnotatedComment(PyccelAstNode):

    """Represents an OpenMP Annotated Comment in the code.

    Parameters
    ----------

    txt: str
        statement to print

    combined: List (Optional)
        constructs to be combined with the current construct

    Examples
    --------
    >>> from pyccel.ast.omp import OmpAnnotatedComment
    >>> OmpAnnotatedComment('parallel')
    OmpAnnotatedComment(parallel)
    """
    __slots__ = ('_txt', '_combined', '_has_nowait')
    _attribute_nodes = ()
    _is_multiline = False

    def __init__(self, txt, has_nowait=False, combined=None):
        self._txt = txt
        self._combined = combined
        self._has_nowait = has_nowait
        super().__init__()

    @property
    def is_multiline(self):
        """Used to check if the construct needs brackets."""
        return self._is_multiline

    @property
    def has_nowait(self):
        """Used to check if the construct has a nowait clause."""
        return self._has_nowait

    @has_nowait.setter
    def has_nowait(self, value):
        """Used to set the _has_nowait var."""
        self._has_nowait = value

    @property
    def name(self):
        """Name of the construct."""
        return ''

    @property
    def txt(self):
        """Used to store clauses."""
        return self._txt

    @property
    def combined(self):
        """Used to store the combined construct of a directive."""
        return self._combined

    def __getnewargs__(self):
        """Used for Pickling self."""
        args = (self.txt, self.combined)
        return args

    def __str__(self):
        instructions = [self.name, self.combined, self.txt]
        return '#$ omp '+' '.join(i for i in instructions if i)

class OMP_For_Loop(OmpAnnotatedComment):
    """ Represents an OpenMP Loop construct. """
    __slots__ = ()
    def __init__(self, txt, has_nowait):
        super().__init__(txt, has_nowait)

    @property
    def name(self):
        """Name of the construct."""
        return 'for'

class OMP_Simd_Construct(OmpAnnotatedComment):
    """ Represents an OpenMP Simd construct"""
    __slots__ = ()
    def __init__(self, txt, has_nowait):
        super().__init__(txt, has_nowait)

    @property
    def name(self):
        """Name of the construct."""
        return 'simd'

class OMP_TaskLoop_Construct(OmpAnnotatedComment):
    """ Represents an OpenMP Taskloop construct"""
    __slots__ = ()
    def __init__(self, txt, has_nowait):
        super().__init__(txt, has_nowait)

    @property
    def name(self):
        """Name of the construct."""
        return 'taskloop'

class OMP_Distribute_Construct(OmpAnnotatedComment):
    """ Represents an OpenMP Distribute construct"""
    __slots__ = ()
    def __init__(self, txt, has_nowait):
        super().__init__(txt, has_nowait)

    @property
    def name(self):
        """Name of the construct."""
        return 'distribute'

class OMP_Parallel_Construct(OmpAnnotatedComment):
    """ Represents an OpenMP Parallel construct. """
    __slots__ = ()
    _is_multiline = True
    @property
    def name(self):
        """Name of the construct."""
        return 'parallel'

class OMP_Task_Construct(OmpAnnotatedComment):
    """ Represents an OpenMP Task construct. """
    __slots__ = ()
    _is_multiline = True
    def __init__(self, txt, has_nowait):
        super().__init__(txt, has_nowait)

class OMP_Single_Construct(OmpAnnotatedComment):
    """ Represents an OpenMP Single construct. """
    __slots__ = ()
    _is_multiline = True
    def __init__(self, txt, has_nowait):
        super().__init__(txt, has_nowait)

    @property
    def name(self):
        """Name of the construct."""
        return 'single'

class OMP_Critical_Construct(OmpAnnotatedComment):
    """ Represents an OpenMP Critical construct. """
    __slots__ = ()
    _is_multiline = True
    def __init__(self, txt, has_nowait):
        super().__init__(txt, has_nowait)

class OMP_Master_Construct(OmpAnnotatedComment):
    """ Represents OpenMP Master construct. """
    __slots__ = ()
    _is_multiline = True
    def __init__(self, txt, has_nowait):
        super().__init__(txt, has_nowait)

class OMP_Masked_Construct(OmpAnnotatedComment):
    """ Represents OpenMP Masked construct. """
    __slots__ = ()
    _is_multiline = True
    @property
    def name(self):
        """Name of the construct."""
        return 'masked'

class OMP_Cancel_Construct(OmpAnnotatedComment):
    """ Represents OpenMP Cancel construct. """
    __slots__ = ()
    def __init__(self, txt, has_nowait):
        super().__init__(txt, has_nowait)

class OMP_Target_Construct(OmpAnnotatedComment):
    """ Represents OpenMP Target construct. """
    __slots__ = ()
    _is_multiline = True
    @property
    def name(self):
        """Name of the construct."""
        return 'target'

class OMP_Teams_Construct(OmpAnnotatedComment):
    """ Represents OpenMP Teams construct. """
    __slots__ = ()
    _is_multiline = True
    @property
    def name(self):
        """Name of the construct."""
        return 'teams'

class OMP_Sections_Construct(OmpAnnotatedComment):
    """ Represents OpenMP Sections construct. """
    __slots__ = ()
    _is_multiline = True
    def __init__(self, txt, has_nowait):
        super().__init__(txt, has_nowait)

    @property
    def name(self):
        """Name of the construct."""
        return 'sections'

class OMP_Section_Construct(OmpAnnotatedComment):
    """ Represent OpenMP Section construct. """
    __slots__ = ()
    _is_multiline = True
    def __init__(self, txt, has_nowait):
        super().__init__(txt, has_nowait)

class Omp_End_Clause(OmpAnnotatedComment):
    """ Represents the End of an OpenMP block. """
    __slots__ = ()
    def __init__(self, txt, has_nowait):
        super().__init__(txt, has_nowait)
