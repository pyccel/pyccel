# pylint: disable=missing-function-docstring, missing-module-docstring

def or_bool(a: "bool", b: "bool"):
    c = False
    if a:
        c = True
    if b:
        c = True
    return c

def real_greater_bool(x0: "float", x1: "float"):
    greater = False
    if x0 > x1:
        greater = True
    return greater

def cast_to_int(a: "float"):
    b = int(a)
    return b

def cast_to_float(a: "int"):
    b = float(a)
    return b

def cast_to_bool(a: "int"):
    b = bool(a)
    return b
