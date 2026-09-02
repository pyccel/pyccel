# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import Final

from pyccel.decorators import allow_negative_index, stack_array


def local_type_annotation():
    gift: int
    gift = 10
    return gift


@allow_negative_index("array")
def allow_negative_index_annotation():
    import numpy as np

    array: "int[:](order=C)"
    array = np.array([1, 2, 3, 4, 5])
    j = -3
    return array[j]


@stack_array("array")
def stack_array_annotation():
    import numpy as np

    array: "int[:,:]"
    array = np.array([[1, 2], [3, 4], [5, 6]])
    return array[2, 0]


def local_type_annotation_2():
    gift: int = 10
    return gift


@allow_negative_index("array")
def allow_negative_index_annotation_2():
    import numpy as np

    array: "int[:](order=C)" = np.array([1, 2, 3, 4, 5])
    j = -3
    return array[j]


@stack_array("array")
def stack_array_annotation_2():
    import numpy as np

    array: "int[:,:]" = np.array([[1, 2], [3, 4], [5, 6]])
    return array[2, 0]


def final_annotation():
    a: Final[int] = 3
    b = a
    return b


def homogeneous_tuple_annotation():
    a: tuple[int, ...]
    a = (1, 2, 3)
    return a[0], a[1], a[2]


def homogeneous_tuple_2_annotation():
    a: tuple[tuple[int, ...], ...]
    a = ((1, 2, 3), (4, 5, 6))
    return a[0][0], a[1][0], a[0][2]


def homogeneous_tuple_annotation_str():
    a: "tuple[int, ...]"
    a = (1, 2, 3)
    return a[0], a[1], a[2]


def homogeneous_tuple_2_annotation_str():
    a: "tuple[tuple[int, ...], ...]"
    a = ((1, 2, 3), (4, 5, 6))
    return a[0][0], a[1][0], a[0][2]


def homogeneous_set_annotation_int():
    a: set[int]
    a = {1, 2, 3, 4}
    return a


def homogeneous_set_without_annotation():
    a = {1, 2, 3, 4}
    return a


def homogeneous_set_annotation_float():
    a: "set[float]"
    a = {1.5, 2.5, 3.3, 4.3}
    return a


def homogeneous_set_annotation_bool():
    a: set[bool]
    a = {False, True, False, True}  # pylint: disable=duplicate-value
    return a


def homogeneous_set_annotation_complex():
    a: "set[complex]"
    a = {1 + 1j, 2 + 2j, 3 + 3j, 1 - 1j}
    return a


def empty_homogeneous_set_annotation_int():
    a: set[int]
    a = set()
    return len(a)


def homogeneous_empty_list_annotation_int():
    a: list[int]
    a = []
    return len(a)


def homogeneous_empty_list_2_annotation_int():
    a: "list[int]"
    a = list()  # pylint: disable=use-list-literal
    return len(a)


def homogeneous_list_annotation_int():
    a: list[int]
    a = [1, 2, 3, 4]
    return a[0], a[1], a[2], a[3]


def homogeneous_list():
    a = [1, 2, 3, 4]
    return a[0], a[1], a[2], a[3]


def homogeneous_list_annotation_float():
    a: list[float]
    a = [1.1, 2.2, 3.3, 4.4]
    return a[0], a[1], a[2], a[3]


def homogeneous_list_annotation_float64():
    from numpy import float64

    a: "list[float64]"
    a = [1.1, 2.2, 3.3, 4.4]
    return a[0], a[1], a[2], a[3]


def homogeneous_list_annotation_bool():
    a: list[bool]
    a = [False, True, True, False]
    return a[0], a[1], a[2], a[3]


def homogeneous_list_annotation_complex():
    a: "list[complex]"
    a = [1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j]
    return a[0], a[1], a[2], a[3]


def dict_int_float():
    a: dict[int, float]
    a = {1: 1.0, 2: 2.0}
    return a


def dict_empty_init():
    a: dict[int, float]
    a = {}
    return a


def dict_complex_float():
    a: dict[complex, float]
    a = {1j: 1.0, -1j: 2.0}
    return a


def inhomogeneous_tuple_annotation():
    a: tuple[int, bool] = (1, True)
    return a[0], a[1]


def inhomogeneous_tuple_annotation_2():
    a: tuple[int] = (1,)
    return a[0]


def inhomogeneous_tuple_annotation_3():
    a: tuple[int, int, int] = (1, 2, 3)
    return a[0], a[1], a[2]


def inhomogeneous_tuple_annotation_4():
    a: tuple[tuple[float, bool], tuple[int, complex]] = ((1.0, False), (1, 2 + 3j))
    return a[0][0], a[0][1], a[1][0], a[1][1]


def inhomogeneous_tuple_annotation_5():
    a: tuple[tuple[int, float]] = ((1, 0.2),)
    return a[0][0], a[0][1]


def inhomogeneous_tuple_annotation_6():
    a: tuple[tuple[tuple[int, float]]] = (((1, 0.2),),)
    return a[0][0][0], a[0][0][1]


def inhomogeneous_tuple_annotation_7():
    a: tuple[tuple[tuple[int, float]], int] = (((1, 0.2),), 1)
    return a[0][0][0], a[0][0][1], a[1]


def inhomogeneous_tuple_annotation_8():
    a: tuple[tuple[tuple[tuple[int, float]], int]] = ((((1, 0.2),), 1),)
    return a[0][0][0][0], a[0][0][0][1], a[0][1]


def str_declaration():
    a: str = (
        "hello here is a very long string with more than 128 characters. This used to be a Fortran limit but now I can hold lots more characters. There is no limit!"
    )
    return len(a)
