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

    def clear(self):
        """
        Remove all items from the dictionary.

        Remove all items from the dictionary. Raises an error.
        """
        raise TypeError("Can't modify read-only dictionary")

    def pop(self, k, d = None, /):
        """
        Pop an item from the dictionary.

        Pop an item from the dictionary if it exists. Else return the default.
        Raises an error.

        Parameters
        ----------
        k : Any
            The key to be searched.
        d : Any
            The default value.
        """
        raise TypeError("Can't modify read-only dictionary")

    def popitem(self):
        """
        Remove and return a key-value pair from the dictionary.

        Remove and return a key-value pair from the dictionary. Raises an error.
        """
        raise TypeError("Can't modify read-only dictionary")

    def update(self, E = None, /, **F):
        """
        Update the dictionary using keywords or another dictionary.

        Update the dictionary using keywords or another dictionary. Raises an error.
        """
        raise TypeError("Can't modify read-only dictionary")
