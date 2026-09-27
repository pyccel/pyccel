# pylint: disable=missing-function-docstring, missing-module-docstring


def swap_basic(a: int, b: int):
    a, b = (b, a)
    return a, b


def swap_basic_2(a: int, b: int):
    a, b = b, a
    return a, b


def swap_basic_3(a: int, b: int, c: int):
    a, b, c = b, c, a
    return a, b, c


def swap_basic_4(a: int, b: int, c: int):
    a, b, c = c, b, a  # pylint: disable=self-assigning-variable
    return a, b, c


def swap_index_1(a: int, b: int, c: int):
    l = [a, b, c]
    l[0], l[1] = l[1], l[0]
    return l[0], l[1], l[2]


def swap_index_2(i: int, j: int):
    l = [1, 2, 3]
    l[i], l[j] = l[j], l[i]
    return l[0], l[1], l[2]


def multi_level_swap(a: int, b: int, c: int):
    d, (b, c) = a, (c, b)
    return a, b, c, d


def multi_type_swap(a: float, b: int, c: float, d: int):
    a, b, c, d = c, d, a, b
    return a, b, c, d


def tuple_assign(a: int, b: int):
    c, d = a, a + b
    return c, d


def tuple_assign2(a: int, b: int):
    a, d = a, a + b  # pylint: disable=self-assigning-variable
    return a, b, d


def tuple_assign3(a: int):
    a, a = a + 3, a + 5  # pylint: disable=redeclared-assigned-name
    return a
