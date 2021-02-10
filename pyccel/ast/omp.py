# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
"""

from .basic import Basic

class OmpAnnotatedComment(Basic):

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

    def __init__(self, txt, combined=None):
        self._txt = txt
        self._combined = combined
        super().__init__()

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

class OMP_For_Loop(OmpAnnotatedComment):
    """ Represents an OpenMP Loop construct. """
    def __init__(self, txt):
        OmpAnnotatedComment.__init__(self, txt)

class OMP_Parallel_Construct(OmpAnnotatedComment):
    """ Represents an OpenMP Parallel construct. """
    def __init__(self, txt, combined=None):
        OmpAnnotatedComment.__init__(self, txt, combined)

class OMP_Task_Construct(OmpAnnotatedComment):
    """ Represents an OpenMP Task construct. """
    def __init__(self, txt):
        OmpAnnotatedComment.__init__(self, txt)

class OMP_Single_Construct(OmpAnnotatedComment):
    """ Represents an OpenMP Single construct. """
    def __init__(self, txt):
        OmpAnnotatedComment.__init__(self, txt)

class OMP_Critical_Construct(OmpAnnotatedComment):
    """ Represents an OpenMP Critical construct. """
    def __init__(self, txt):
        OmpAnnotatedComment.__init__(self, txt)

class OMP_Master_Construct(OmpAnnotatedComment):
    """ Represents OpenMP Master construct. """
    def __init__(self, txt):
        OmpAnnotatedComment.__init__(self, txt)

class OMP_Masked_Construct(OmpAnnotatedComment):
    """ Represents OpenMP Masked construct. """
    def __init__(self, txt, combined=None):
        OmpAnnotatedComment.__init__(self, txt, combined)

class OMP_Cancel_Construct(OmpAnnotatedComment):
    """ Represents OpenMP Cancel construct. """
    def __init__(self, txt):
        OmpAnnotatedComment.__init__(self, txt)

class OMP_Target_Construct(OmpAnnotatedComment):
    """ Represents OpenMP Target construct. """
    def __init__(self, txt, combined=None):
        OmpAnnotatedComment.__init__(self, txt, combined)

class OMP_Teams_Construct(OmpAnnotatedComment):
    """ Represents OpenMP Target construct. """
    def __init__(self, txt, combined=None):
        OmpAnnotatedComment.__init__(self, txt, combined)

class OMP_Sections_Construct(OmpAnnotatedComment):
    """ Represents OpenMP Sections construct. """
    def __init__(self, txt):
        OmpAnnotatedComment.__init__(self, txt)

class OMP_Section_Construct(OmpAnnotatedComment):
    """ Represent OpenMP Section construct. """
    def __init__(self, txt):
        OmpAnnotatedComment.__init__(self, txt)

class Omp_End_Clause(OmpAnnotatedComment):
    """ Represents the End of an OpenMP block. """
    def __init__(self, txt):
        OmpAnnotatedComment.__init__(self, txt)
