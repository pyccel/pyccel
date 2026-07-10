# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import TypeVar, Final

def add_int():
    a = {1, 3, 45}
    a.add(4)
    return len(a)

def add_complex():
    a = {6j, 7j, 8j}
    a.add(9j)
    return len(a)

def add_element_range():
    a = {1, 2, 3}
    for i in range(50, 100):
        a.add(i)
    return len(a)

def clear_int():
    se = {1, 2, 4, 5}
    se.clear()
    return len(se)

def clear_float():
    se = {7.2, 2.1, 9.8, 6.4}
    se.clear()
    return len(se)

def clear_complex():
    se = {3j, 6j, 2j}
    se.clear()
    return len(se)

def copy_int():
    se = {1, 2, 4, 5}
    cop = se.copy()
    size = len(cop)
    a, b, c, d = cop.pop(), cop.pop(), cop.pop(), cop.pop()
    return size, len(se), a, b, c, d

def copy_float():
    se = {5.7, 6.2, 4.3, 9.8}
    cop = se.copy()
    return len(cop), cop.pop(), cop.pop(), cop.pop(), cop.pop(), len(se)

def copy_complex():
    se = {7j, 6j, 9j}
    cop = se.copy()
    return len(cop), cop.pop(), cop.pop(), cop.pop()

def remove_complex():
    se = {1j, 3j, 8j}
    se.remove(3j)
    return se

def remove_int():
    se = {2, 4, 9}
    se.remove(4)
    return se

def remove_float():
    se = {5.7, 2.4, 8.1}
    se.remove(8.1)
    return se

def Discard_int():
    se = {2.7, 4.3, 9.2}
    se.discard(4.3)
    return se

def Discard_complex():
    se = {2j, 5j, 3j, 7j}
    se.discard(5j)
    return se

def Discard_wrong_arg():
    se = {4.7, 1.3, 8.2}
    se.discard(8.6)
    return se

def update_basic():
    a = {1, 2, 3}
    b = {4, 5, 6}
    a.update(b)
    return len(a), a.pop(), a.pop(), a.pop(), a.pop(), a.pop(), a.pop()

def update_multiple():
    a = {1, 2, 3}
    a.update({4, 5})
    a.update({6, 7, 8, 9})
    a.update({10})
    return (
        len(a),
        a.pop(),
        a.pop(),
        a.pop(),
        a.pop(),
        a.pop(),
        a.pop(),
        a.pop(),
        a.pop(),
        a.pop(),
        a.pop(),
    )

def update_multiple_args():
    a = {1, 2, 3}
    a.update({4, 5}, {6, 7, 8, 9}, {10})
    return (
        len(a),
        a.pop(),
        a.pop(),
        a.pop(),
        a.pop(),
        a.pop(),
        a.pop(),
        a.pop(),
        a.pop(),
        a.pop(),
        a.pop(),
    )

def update_boolean_tuple():
    a = {True}
    b = (False, True, False)
    a.update(b)
    return len(a), a.pop(), a.pop()

def update_complex_list():
    a = {1j, 2 + 3j, 0 + 0j}
    b = [4j, 5j, 1 + 6j]
    a.update(b)
    return len(a), a.pop(), a.pop(), a.pop(), a.pop(), a.pop(), a.pop()

def update_range():
    a = {1, 2, 3}
    a.update(range(4, 9))
    return (
        len(a),
        a.pop(),
        a.pop(),
        a.pop(),
        a.pop(),
        a.pop(),
        a.pop(),
        a.pop(),
        a.pop(),
    )

def update_set_as_arg():
    a = {1, 2, 3}
    a.update({4, 5, 6})
    return len(a), a.pop(), a.pop(), a.pop(), a.pop(), a.pop(), a.pop()

def update_tuple_as_arg():
    a = {1, 2, 3}
    a.update((4, 5, 6))
    return len(a), a.pop(), a.pop(), a.pop(), a.pop(), a.pop(), a.pop()

def set_With_list():
    a = [1.6, 6.3, 7.2]
    b = set(a)
    return b

def set_With_tuple():
    a = (1j, 6j, 7j)
    b = set(a)
    return b

def set_With_set():
    a = {True, False, True}  # pylint: disable=duplicate-value
    b = set(a)
    return b

def init_with_set():
    b = set({4.6, 7.9, 2.5})
    return len(b), b.pop(), b.pop(), b.pop()

def init_with_list():
    b = set([4.6, 7.9, 2.5])
    return len(b), b.pop(), b.pop(), b.pop()

def copy_from_arg2(a: "set[float]"):
    b = set(a)
    return b

def Pop_int():
    se = {2, 4, 9}
    el1 = se.pop()
    el2 = se.pop()
    el3 = se.pop()
    return el1, el2, el3

def Pop_float():
    se = {2.3, 4.1, 9.5}
    el1 = se.pop()
    el2 = se.pop()
    el3 = se.pop()
    return el1, el2, el3

def Pop_complex():
    se = {4j, 1j, 7j}
    el1 = se.pop()
    el2 = se.pop()
    el3 = se.pop()
    return el1, el2, el3

def union_int():
    a = {1, 2, 3, 4}
    b = {5, 6, 7, 2}
    c = a.union(b)
    return a, b, c

def set_union_no_args():
    a = {1, 2, 3, 4}
    c = a.union()
    a.add(5)
    return len(c), c.pop(), c.pop(), c.pop(), c.pop()

def set_union_2_args():
    a = {1, 2, 3, 4}
    b = {5, 6, 7}
    c = {8, 9, 10, 4}
    d = a.union(b, c)
    return (
        len(d),
        d.pop(),
        d.pop(),
        d.pop(),
        d.pop(),
        d.pop(),
        d.pop(),
        d.pop(),
        d.pop(),
        d.pop(),
        d.pop(),
    )

def set_union_temporaries():
    c = {1, 2, 3, 4}.union({5, 6, 7, 2})
    return len(c), c.pop(), c.pop(), c.pop(), c.pop(), c.pop(), c.pop(), c.pop()

def temporary_set_union_2():
    a = [1, 2]
    b = {2}
    d = set(a).union(b)
    return d

def union_list():
    a = {1.2, 2.3}
    b = [1.2, 5.0]
    d = a.union(b)
    return len(d), d.pop(), d.pop(), d.pop()

def union_tuple():
    a = {True}
    b = (False,)
    d = a.union(b)
    return len(d), d.pop(), d.pop()

def set_union_operator():
    a = {1, 2, 3, 4}
    b = {5, 6, 7, 2}
    c = a | b
    return len(c), c.pop(), c.pop(), c.pop(), c.pop(), c.pop(), c.pop(), c.pop()

def set_union_augoperator():
    a = {1, 2, 3, 4}
    b = {5, 6, 7, 2}
    a |= b
    return len(a), a.pop(), a.pop(), a.pop(), a.pop(), a.pop(), a.pop(), a.pop()

def intersection_int():
    a = {1, 2, 3}
    b = {2, 3, 4}
    c = a.intersection(b)
    return len(c), c.pop(), c.pop()

def set_intersection_no_args():
    a = {1, 2, 3, 4}
    c = a.intersection()
    a.add(5)
    return len(c), c.pop(), c.pop(), c.pop(), c.pop()

def set_intersection_2_args():
    a = {1, 2, 3, 4}
    b = {5, 6, 7, 2, 1, 3}
    c = {7, 6, 10, 4, 2, 3, 1}
    d = a.intersection(b, c)
    return len(d), d.pop(), d.pop(), d.pop()

def set_intersection_int_temporaries():
    c = {1, 2, 3}.intersection({2, 3, 4})
    return len(c), c.pop(), c.pop()

def temporary_set_intersection():
    a = {1, 2}
    b = {2}
    d = a.intersection(b).pop()
    return d

def set_intersection_operator():
    a = {1, 2, 3, 4, 8}
    b = {5, 2, 3, 7, 8}
    c = a & b
    return len(c), c.pop(), c.pop(), c.pop()

def set_intersection_operator_2():
    a = {1, 2, 3, 4, 8}
    b = {5, 2, 3, 7, 8}
    return a & b

def set_intersection_update():
    a = {1, 2, 3, 4, 8}
    b = {5, 2, 3, 7, 8}
    a.intersection_update(b)
    return len(a), a.pop(), a.pop(), a.pop()

def set_intersection_multiple_update():
    a = {1, 2, 3, 4, 8}
    b = {5, 2, 3, 7, 8}
    c = {10, 2, 20}
    a.intersection_update(b, c)
    return len(a), a.pop()

def set_intersection_augoperator():
    a = {1, 2, 3, 4}
    b = {2, 3, 4}
    a &= b
    return len(a), a.pop(), a.pop(), a.pop()

def set_contains():
    a = {1, 2, 3, 4, 5, 6, 7, 8}
    b = 2 in a
    return b, (4 in a), (9 in a)

def set_ptr():
    a = {1, 2, 3, 4, 5, 6, 7, 8}
    b = a
    b.pop()
    return len(a), len(b)

def set_sum_int():
    a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 12}
    sum_a = 0
    for ai in a:
        sum_a += ai
    return sum_a

def set_iter_prod():
    # Integers must be used to get an exact result to compare sets
    from itertools import product

    a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 12}
    b = {2, 4, 9, 1, 8}
    assemble = 0.0
    for ai, bi in product(a, b):
        assemble += ai * bi
    return assemble

T_set_arg = TypeVar("T_set_arg", int, float, complex)

def set_arg(arg: Final[set[T_set_arg]], my_sum: T_set_arg):
    for ai in arg:
        my_sum += ai
    return my_sum

def set_return():
    a = {1, 2, 3, 4, 5}
    return a

def set_min_max():
    a_int = {1, 2, 3, 4, 5}
    a_float = {1.1, 2.2, 3.3, 4.4, 5.5}
    return min(a_int), min(a_float), max(a_int), max(a_float)

def set_is_disjoint(a: set[int], b: set[int]):
    return a.isdisjoint(b)

def difference_int():
    a = {1, 2, 3}
    b = {2, 3, 4}
    c = a.difference(b)
    return len(c), c.pop()

def set_difference_no_args():
    a = {1, 2, 3, 4}
    c = a.difference()
    a.add(5)
    return len(c), c.pop(), c.pop(), c.pop(), c.pop()

def set_difference_2_args():
    a = {1, 2, 3, 4}
    b = {5, 6, 7, 2, 1}
    c = {7, 6, 10, 2, 3, 1}
    d = a.difference(b, c)
    return len(d), d.pop()

def set_difference_int_temporaries():
    c = {1, 2, 3}.difference({3, 4})
    return len(c), c.pop(), c.pop()

def temporary_set_difference():
    a = {1, 2}
    b = {2}
    d = a.difference(b).pop()
    return d

def set_difference_operator():
    a = {1, 2, 3, 4, 8}
    b = {5, 2, 3, 7, 8}
    c = a - b
    return len(c), c.pop(), c.pop()

def set_difference_update():
    a = {1, 2, 3, 4, 8}
    b = {5, 2, 3, 7, 8}
    a.difference_update(b)
    return len(a), a.pop(), a.pop()

def set_difference_multiple_update():
    a = {1, 2, 3, 4, 8}
    b = {5, 2, 3, 7, 8}
    c = {10, 4, 20}
    a.difference_update(b, c)
    return len(a), a.pop()

def set_difference_augoperator():
    a = {1, 2, 3, 4}
    b = {2, 3, 4}
    a -= b
    return len(a), a.pop()
