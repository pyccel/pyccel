# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import TypeVar


def sum_range(a0: "int[:]"):
    return sum(a0[i] for i in range(len(a0)))


def sum_var(a: "int[:]"):
    return sum(ai for ai in a)


def sum_var2(a: "int[:,:]"):
    return sum(aii for ai in a for aii in ai)


def sum_var3(a: "int[:,:,:]"):
    m, n, p = a.shape
    return sum(a[i, j, k] for i in range(m) for j in range(n) for k in range(p))


def sum_var4(a: "int[:]"):
    s = 3
    return sum(ai for ai in a), s


def sum_var5(a: "bool[:]"):
    return sum(ai for ai in a)


def expression1(b: "float[:]"):
    n = b.shape[0]
    return (2 * sum(b[i] for i in range(n)) ** 5 + 5) * min(
        j + 1.0 for j in b
    ) ** 4 + 9 * max(j + 1.0 for j in b) ** 4


def expression2(b: "int64[:]"):
    def incr(x: "int64"):
        y = x + 1
        return y

    n = b.shape[0]
    return 5 + incr(2 + incr(6 + sum(b[i] for i in range(n))))


def nested_generators1(a: "float[:,:,:,:]"):
    return sum(
        sum(sum(a[i, k, o, 2] for i in range(5)) for k in range(5)) for o in range(5)
    )


def nested_generators2(a: "float[:,:,:,:]"):
    return min(
        min(
            sum(
                min(max(a[i, k, o, l] * l for i in range(5)) for k in range(5))
                for o in range(5)
            )
            for l in range(5)
        ),
        0.0,
    )


def nested_generators3(a: "float[:,:,:,:]"):
    return sum(sum(a[i, k, 4, 2] for i in range(5)) for k in range(5)) ** 2


def nested_generators4(a: "float[:,:,:,:]"):
    return min(max(a[i, k, 4, 2] for i in range(5)) for k in range(5)) ** 2


def sum_range_overwrite(a0: "int[:]"):
    v = sum(a0[i] for i in range(len(a0)))
    v = sum(a0[i] for i in range(len(a0)))
    return v


def sum_with_condition():
    v = sum(i for i in range(20) if i % 2 == 1)
    return v


def sum_with_multiple_conditions():
    v = sum(i - j for i in range(20) if i % 2 == 1 for j in range(30) if j % 3 == 1)
    return v


def max_with_condition():
    v = max(i for i in range(20) if i % 2 == 1)
    return v


def max_with_condition_float():
    v = max(i / 2 for i in range(20) if i % 2 == 1)
    return v


def max_with_multiple_conditions():
    v = max(i - j for i in range(20) if i % 2 == 1 for j in range(30) if j % 3 == 1)
    return v


def min_with_condition():
    v = min(i for i in range(20) if i % 2 == 1)
    return v


def min_with_condition_float():
    v = min(i / 2 for i in range(20) if i % 2 == 1)
    return v


def min_with_multiple_conditions():
    v = min(i - j for i in range(20) if i % 2 == 1 for j in range(30) if j % 3 == 1)
    return v


def sum_with_two_variables():
    x = sum(i - j for i in range(10) for j in range(7))
    return x


T_min_max_values_f = TypeVar(
    "T_min_max_values_f", "int16[:]", "int32[:]", "int64[:]", "float32[:]", "float64[:]"
)


def min_max_values(a: T_min_max_values_f):
    min_val = min(ai for ai in a)
    max_val = max(ai for ai in a)
    return min_val, max_val
