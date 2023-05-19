# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
""" Module containing helper functions for managing strings
"""

def create_incremented_string(forbidden_exprs, prefix = 'Dummy', counter = 1, name_clash_checker = None):
    """
    Create a new unique string by incrementing a prefix.

    This function takes a prefix and a counter and uses them to construct
    a new name of the form:
            prefix_counter
    Where counter is formatted to fill 4 characters
    The new name is checked against a list of forbidden expressions. If the
    constructed name is forbidden then the counter is incremented until a valid
    name is found.

    Parameters
    ----------
    forbidden_exprs : set
        A set of all the values which are not valid solutions to this problem.
    prefix : str
        The prefix used to begin the string.
    counter : int
        The expected value of the next name.
    name_clash_checker : LanguageNameClashChecker
        A class instance providing access to a `has_clash` function which determines
        if names clash in a given language.

    Returns
    -------
    name : str
        The incremented string name.
    counter : int
        The expected value of the next name.
    """
    nDigits = 4

    if prefix is None:
        prefix = 'Dummy'

    name_format = "{prefix}_{counter:0="+str(nDigits)+"d}"
    name = name_format.format(prefix=prefix, counter = counter)
    counter += 1
    if name_clash_checker:
        while name_clash_checker.has_clash(name, forbidden_exprs):
            name = name_format.format(prefix=prefix, counter = counter)
            counter += 1
    else:
        while name in forbidden_exprs:
            name = name_format.format(prefix=prefix, counter = counter)
            counter += 1

    return name, counter
