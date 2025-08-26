# pylint: disable=missing-function-docstring, missing-module-docstring

def functional_for_1d_range():
    a = [i+1 for i in range(4)]
    return len(a), a[0], a[1], a[2], a[3]

def functional_for_overwrite_1d_range():
    a = [i+1 for i in range(4)]
    a = [i+1 for i in range(1,5)]
    return len(a), a[0], a[1], a[2], a[3]

def functional_for_1d_var(y : 'int[:]'):
    a = [yi+1 for yi in y]
    return len(a), a[0], a[1], a[2], a[3]

def functional_for_1d_const(y : 'int[:]', z : 'int'):
    a = [z for _ in y]
    return len(a), a[0], a[1], a[2], a[3]

def functional_for_1d_const2():
    a = [5 for _ in range(0,4,2)]
    return len(a), a[0], a[1]

def functional_for_2d_range():
    a = [i*j for i in range(3) for j in range(2)]
    return len(a), a[0], a[1], a[2], a[3], a[4], a[5]

def functional_for_2d_var_range(y : 'int[:]'):
    a = [yi for yi in y for j in range(2)]
    return len(a), a[0], a[1], a[2], a[3], a[4], a[5]

def functional_for_2d_var_var(y : 'int[:]', z : 'int[:]'):
    a = [yi*zi for yi in y for zi in z]
    return len(a), a[0], a[1], a[2], a[3], a[4], a[5]

def functional_for_2d_dependant_range_1():
    a = [i*j for i in range(4) for j in range(i)]
    return len(a), a[0], a[1], a[2], a[3], a[4], a[5]

def functional_for_2d_dependant_range_2():
    a = [i*j for i in range(3) for j in range(i,3)]
    return len(a), a[0], a[1], a[2], a[3], a[4], a[5]

def functional_for_2d_dependant_range_3():
    a = [i*j for i in range(1,4) for j in range(0,4,i)]
    return len(a), a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]

def functional_for_2d_array_range(idx : 'int'):
    a = [(x1, y1, z1)  for x1 in range(3) for y1 in range(x1,5) for z1 in range(y1,10)]
    return len(a), a[idx][0], a[idx][1], a[idx][2]

def functional_for_2d_range_const():
    a = [20 for _ in range(3) for _ in range(2)]
    return len(a), a[0], a[1], a[2], a[3], a[4], a[5]

def functional_for_3d_range():
    a = [i*j for i in range(1,3) for j in range(i,4) for k in range(i,j)]
    return len(a), a[0], a[1], a[2], a[3]

def unknown_length_functional(x : 'int[:]'):
    a = [i*3 for i in range(len(x)*2)]
    return len(a), a[0], a[-1]

def functional_with_enumerate():
    a = [x + 1 for x in range(10)]
    b = [i * j for i,j in enumerate(a)]
    return len(b), b[0], b[1], b[2]

def functional_with_enumerate_with_start():
    a = [x - 3  for x in range(10)]
    b = [i * j for i,j in enumerate(a, start=5)]
    return len(b), b[0], b[1], b[2]

def functional_with_condition():
    a = [x if x %2 == 0 else -x for x in range(30)]
    return len(a), a[0], a[1], a[2], a[3]

def functional_with_zip():
    a = [x ** 2 for x in range(8)]
    b = [0, 1, 2]
    c = [k-y for k,y in zip(a,b)]
    return len(c), c[0], c[1], c[2]

def functional_with_multiple_zips():
    a = [x ** 2 for x in range(8)]
    b = [0, 1, 2]
    c = [y + 1 for y in range(10)]
    d = [i + j + k for i, j, k in zip(a, b, c)]
    return len(d), d[0], d[1], d[2]

def functional_filter_and_transform():
    a = [x + 1 for x in range(10) if x % 2 == 0]
    return len(a), a[0], a[1], a[2]


def functional_with_multiple_conditions():
    a = [x + 1 - y for x in range(10) if x % 2 == 0 for y in range(12) if y % 3 == 0]
    return len(a), a[0], a[1], a[2]

def functional_negative_indices(arg : 'int[:]'):
    a = [ai*i for i, ai in enumerate(arg[-5:-1])]
    return len(a), a[0], a[1], a[2], a[3]

def functional_reverse(arg : 'int[:]'):
    a = [ai for ai in arg[::-1]]
    return len(a), a[0], a[1], a[2], a[3]
