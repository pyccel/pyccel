# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
Handles name clash problems in Python
"""
from pyccel.utilities.metaclasses import Singleton
from pyccel.utilities.strings import create_incremented_string
from .languagenameclashchecker import LanguageNameClashChecker

class PythonNameClashChecker(LanguageNameClashChecker):
    """ Class containing functions to help avoid problematic names in Python
    """
    keywords = set()

    def has_clash(self, name, symbols):
        """ Indicate whether the proposed name causes any clashes
        """
        return any(name == k for k in self.keywords) or \
               any(name == s for s in symbols)

    def get_collisionless_name(self, name, symbols):
        """
        Get a valid name which doesn't collision with symbols.

        Find a new name based on the suggested name which does not
        appear in the provided symbols. It is not necessary to exclude
        keywords for names which were either originally valid Python
        names, or internally generated names.

        Parameters
        ----------
        name : str
            The suggested name.
        symbols : set
            Symbols which should be considered as collisions.

        Returns
        -------
        str
            A new name which is collision free.
        """
        return self._get_collisionless_name(name, symbols)

