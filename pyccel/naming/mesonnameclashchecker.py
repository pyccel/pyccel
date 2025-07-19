# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
Handles name clash problems in CMake. 
"""
from .languagenameclashchecker import LanguageNameClashChecker

class MesonNameClashChecker(LanguageNameClashChecker):
    """
    Class containing functions to help avoid problematic names in meson.

    A class which provides functionalities to check or propose variable names and
    verify that they do not cause name clashes. Name clashes may be due to
    new variables, or due to the use of reserved keywords.
    """
    keywords = set(['library', 'link_with', 'dependencies', 'install', 'install_dir',
                    'executable', 'true', 'false', 'subdir', 'project', 'import',
                    'dependency'])

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

    def get_collisionless_name(self, name, symbols):
        """
        Get a valid name which doesn't collision with symbols or meson keywords.

        Find a new name based on the suggested name which will not cause
        conflicts with meson keywords, does not appear in the provided symbols,
        and is a valid name in meson code.

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

