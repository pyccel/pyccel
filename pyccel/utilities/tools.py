#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
File containing basic tools.
"""

__all__ = (
        'ReadOnlyDict',
        )

class ReadOnlyDict(dict):
    """
    A read-only version of a dictionary.

    A read-only version of a dictionary. This is useful to prevent objects
    being used in a way which was not intended.
    """
    def __setitem__(self, key, value):
        raise TypeError("Can't modify read-only dictionary")

    def clear(self, *args):
        raise TypeError("Can't modify read-only dictionary")

    def pop(self, *args):
        raise TypeError("Can't modify read-only dictionary")

    def popitem(self, *args):
        raise TypeError("Can't modify read-only dictionary")

    def update(self, *args):
        raise TypeError("Can't modify read-only dictionary")
