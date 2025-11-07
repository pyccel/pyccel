# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
Handles name clash problems in Python
"""
from .languagenameclashchecker import LanguageNameClashChecker

class PythonNameClashChecker(LanguageNameClashChecker):
    """
    Class containing functions to help avoid problematic names in Python.

    A class which provides functionalities to check or propose variable names and
    verify that they do not cause name clashes. Name clashes may arise when
    generating names for new variables.
    """
    keywords = set()

    def has_clash(self, name, symbols):
        """
        Indicate whether the proposed name causes any clashes.

        Indicate whether the proposed name causes any clashes by comparing it with the
        reserved keywords and the symbols which are already defined in the scope.

        Parameters
        ----------
        name : str
            The proposed name.
        symbols : set of str
            The symbols already used in the scope.

        Returns
        -------
        bool
            True if the name clashes with an existing name. False otherwise.
        """
        return name in self.keywords or name in symbols

    def get_collisionless_name(self, name, symbols, *, prefix, context, parent_context):
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
        prefix : str
            The prefix that may be added to the name to provide context information.
        context : str
            The context where the name will be used.
        parent_context : str
            The type of the scope where the object with this name will be saved.

        Returns
        -------
        str
            A new name which is collision free.
        """
        assert context in ('module', 'function', 'class', 'variable')
        assert parent_context in ('module', 'function', 'class', 'loop', 'program')
        return self._get_collisionless_name(name, symbols)

