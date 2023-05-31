#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
""" Module containing metaclasses which are useful for the rest of pyccel
"""


__all__ = (
    'Singleton',
    'ArgumentSingleton',
)

class Singleton(type):
    """
    Metaclass indicating that there is only one instance of the class.

    A metaclass which ensures that only one instance of the class is ever
    created. Trying to create a second instance will result in accessing
    the first.
    """
    _instances = {}
    def __call__(cls):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__()
        return cls._instances[cls]

class ArgumentSingleton(type):
    """ Indicates that there is only one instance of the class
    for any given set of arguments
    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        index = (cls, *args, *sorted(kwargs.items()))
        if index not in cls._instances:
            cls._instances[index] = super().__call__(*args, **kwargs)
        return cls._instances[index]
