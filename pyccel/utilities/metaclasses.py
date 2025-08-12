#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
""" Module containing metaclasses which are useful for the rest of pyccel
"""
from inspect import signature

__all__ = (
    'Singleton',
    'ArgumentSingleton',
)


class ArgumentSingleton(type):
    """
    Metaclass indicating that there is only one instance of the parametrised class.

    Metaclass indicating that there is only one instance of the class for any given
    set of arguments.

    Parameters
    ----------
    name : str
        The name of the class.
    bases : tuple[class,...]
        A tuple of the superclasses of the class.
    dct : dict
        A dictionary of the class attributes.
    """
    def __init__(cls, name, bases, dct):
        cls._instances = {}
        # Trick inspect.signature into seeing the signature of
        # cls.__init__ so numpydoc checks the correct signature
        cls.__signature__ = signature(cls.__init__)
        super().__init__(name, bases, dct)

    def __call__(cls, *args, **kwargs):
        index = (*args, *sorted(kwargs.items()))
        existing_instance = next((i for i in cls._instances if i is index), None)
        if existing_instance is None:
            new_instance = super().__call__(*args, **kwargs)
            cls._instances[index] = new_instance
            return new_instance
        else:
            return existing_instance

class Singleton(type):
    """
    Metaclass indicating that there is only one instance of the class.

    A metaclass which ensures that only one instance of the class is ever
    created. Trying to create a second instance will result in accessing
    the first.

    Parameters
    ----------
    name : str
        The name of the class.
    bases : tuple[class,...]
        A tuple of the superclasses of the class.
    dct : dict
        A dictionary of the class attributes.
    """
    def __init__(cls, name, bases, dct):
        cls._instance = None
        # Trick inspect.signature into seeing the signature of
        # cls.__init__ so numpydoc checks the correct signature
        cls.__signature__ = signature(cls.__init__)
        super().__init__(name, bases, dct)

    def __call__(cls):
        existing_instance = cls._instance
        if existing_instance is None:
            new_instance = super().__call__()
            cls._instance = new_instance
            return new_instance
        else:
            return existing_instance
