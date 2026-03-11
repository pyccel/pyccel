# pylint: disable=missing-function-docstring, missing-module-docstring
def g(x: float, v: float) -> tuple[float, float]:
    m = x - v
    t = 2.0 * m
    z = 2.0 * t
    return t, z


z = 0.0
t = 0.0

# ...
xd = 5.0
w = 5.0
[z, t] = g(xd, w)
