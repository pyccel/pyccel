#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
""" Module containing metaclasses which are useful for the rest of pyccel
"""


__all__ = (
    'Singleton',
    'build_argument_singleton'
)

def build_argument_singleton(*argnames):
    """
    Build a metaclass which handles singletons with arguments.

    Build a metaclass which ensures that there is only one instance of the class
    for any given set of arguments. The metaclass is specialised for the
    specified arguments which ensures that `inspect.signature` returns the
    expected result and that the docstring can match the `__init__` method.

    Parameters
    ----------
    *argnames : tuple[str]
        A tuple of strings describing all the arguments which may be passed to
        the function.

    Returns
    -------
    type
        A metaclass which can be used when building classes.

    Examples
    --------
    >>> class A(metaclass = build_argument_singleton('arg1', 'arg2 = 3')):
            def __init__(self, arg1, arg2 = 3):
                pass
    """
    args = ', '.join([*argnames])
    def_code = '\n'.join([f"def new_call_func({args}):",
                           "    index = (cls, {args})",
                           "    if index not in cls._instances:",
                           "        cls._instances[index] = super().__call__(*args, **kwargs)",
                           "    return cls._instances[index]"])
    new_call_func = exec(def_code)
    return type('ArgumentSingleton', (type,),
            {'__call__': new_call_func,
             '_instances': {}})

class Singleton(ArgumentSingleton):
    """
    Metaclass indicating that there is only one instance of the class.

    A metaclass which ensures that only one instance of the class is ever
    created. Trying to create a second instance will result in accessing
    the first.
    """
    def __call__(cls):
        return super().__call__()
