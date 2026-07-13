# pylint: disable=missing-function-docstring, missing-module-docstring

def call_gcd(x: int, y: int):
    from math import gcd

    return gcd(x, y)

def call_factorial(x: "int"):
    from math import factorial

    return factorial(x)

def call_lcm(x: int, y: int):
    from math import lcm

    return lcm(x, y)

def call_radians(x: "float"):
    from math import radians

    return radians(x)

def call_degrees(x: "float"):
    from math import degrees

    return degrees(x)

def call_degrees_i(x: "int"):
    from math import degrees

    return degrees(x)
