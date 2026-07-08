# pylint: disable=missing-function-docstring, missing-module-docstring
"""
Functions used to test list support, block-translated as a single module
via the `lists_mod` fixture in test_epyccel_lists.py.
"""

from typing import Final, TypeVar

T = TypeVar("T", int, float, complex)


def pop_last_element():
    a = [1, 3, 45]
    return a.pop()


def pop_list_bool():
    a = [True, False, True]
    return a.pop()


def pop_list_float():
    a = [1.5, 3.1, 4.5]
    return a.pop()


def pop_specific_index():
    a = [1j, 3j, 45j]
    return a.pop(1)


def pop_negative_index():
    a = [1j, 3j, 45j]
    return a.pop(-1)


def pop_2():
    a = [1.7, 2.7, 45.0]
    a.pop()
    return a.pop(-1)


def pop_expression():
    a = [1, 3, 45]
    return a.pop() + 3


def pop_as_arg():
    a = [1, 3, 45]
    return a.pop(a.pop(0))


def append_basic():
    a = [1, 2, 3]
    a.append(4)
    return len(a), a[0], a[1], a[2], a[3]


def append_multiple():
    a = [1, 2, 3]
    a.append(4)
    a.append(5)
    a.append(6)
    return len(a), a[0], a[1], a[2], a[3], a[4], a[5]


def append_range():
    a = [1, 2, 3]
    for i in range(0, 1000):
        a.append(i)
    a.append(1000)
    return len(a), a[-1], a[-2]


def append_bool():
    a = [True, True, True]
    a.append(False)
    a.append(False)
    a.append(True)
    return len(a), a[3], a[4], a[5]


def append_float():
    a = [3.5, 2.2, 1.5]
    a.append(3.0)
    a.append(2.9)
    a.append(1.1)
    return len(a), a[3], a[4], a[5]


def append_complex():
    a = [1 + 2j, 3 + 4j, 5 + 6j]
    a.append(9j)
    a.append(2 + 2j)
    a.append(1j)
    return len(a), a[3], a[4], a[5]


def insert_booleans():
    a = [True, False, True]
    a.insert(0, True)
    a.insert(-100, True)
    a.insert(1000, False)
    a.insert(0, False)
    a.insert(666, True)
    a.insert(-1, True)
    a.insert(-25, False)
    return a


def insert_complex():
    a = [2j, 3 + 6j, 0 + 0j]
    a.insert(0, 9j)
    a.insert(-100, 1 - 1j)
    a.insert(1000, -3j)
    a.insert(0, 0j)
    a.insert(666, 1j)
    a.insert(-1, 1 + 0j)
    a.insert(-25, 0 - 0j)
    return a


def insert_float():
    a = [0.0, 3.6, 0.5]
    a.insert(0, 6.4)
    a.insert(-100, 25.12)
    a.insert(1000, 13.04)
    a.insert(0, 19.99)
    a.insert(666, 20.00)
    a.insert(-1, 3.01)
    a.insert(-25, 2.5)
    return a


def insert_multiple():
    a = [1, 2, 3]
    a.insert(4, 4)
    a.insert(2, 5)
    a.insert(1, 6)
    return a


def insert_range():
    a = [1, 2, 3]
    for i in range(4, 1000):
        a.insert(i - 1, i)
    return a


def clear_1():
    a = [1, 2, 3]
    a.clear()
    return a


def clear_2():
    a: "list[int]" = []
    a.clear()
    return a


def list_contains():
    a = [1, 3, 4, 7, 10, 3]
    return (1 in a), (5 in a), (3 in a)


def list_ptr():
    a = [1, 3, 4, 7, 10, 3]
    b = a
    b.append(22)
    return len(a), len(b)


def list_return():
    a = [1, 2, 3, 4, 5]
    return a


def list_min_max():
    a_int = [1, 2, 3, 4]
    a_float = [1.1, 2.2, 3.3, 4.4]
    return min(a_int), max(a_int), min(a_float), max(a_float)


def list_reverse():
    a_int = [1, 2, 3]
    a_float = [1.1, 2.2, 3.3]
    a_complex = [1j, 2 - 3j]
    a_single = [1]
    a_int.reverse()
    a_float.reverse()
    a_complex.reverse()
    a_single.reverse()
    return (
        a_int[0],
        a_int[-1],
        a_float[0],
        a_float[-1],
        a_single[0],
        a_single[-1],
        a_complex[0],
        a_complex[-1],
    )


def list_const_arg(arg: Final[list[T]], my_sum: T):
    for ai in arg:
        my_sum += ai
    return my_sum


def list_equality(arg1: Final[list[int]], arg2: Final[list[int]]):
    return arg1 == arg2


def list_duplicate(n: int):
    a = [1] * n
    b = [1, 2, 3] * n
    return a, b


def list_assign_slices(a: Final[list[T]]):
    N: Final[int] = len(a)
    O: Final[T] = a[0] - a[0]
    b: list[T] = [O] * N
    c: list[T] = [O] * (N - 1)
    d: list[T] = [O] * ((N + 1) // 2)
    e: list[T] = [O] * ((N - 1) // 2)
    b[:] = a[-1::-1]
    c[:] = a[1:]
    d[:] = a[::2]
    e[:] = a[-2::-2]
    return b, c, d, e
