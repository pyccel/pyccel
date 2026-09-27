# pylint: disable=missing-function-docstring, missing-module-docstring


def n1_line_docstring():
    """short doc string"""
    return 1


def multiline_line_docstring():
    """
    Big beautiful doc string

    Parameters
    ----------

    Results
    -------
    1 : int
        no description
    """
    return 1


class MyClass:
    """
    Empty class
    """

    def __init__(self: "MyClass"):
        pass


class MyClassProperty:
    """
    Class containing x
    """

    def __init__(self: "MyClassProperty", x: int):
        self._x = x

    @property
    def x(self):
        """
        This is a property it cannot be set.
        """
        return self._x
