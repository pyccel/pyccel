# pylint: disable=missing-function-docstring, missing-module-docstring

#$ header template S(int|real)
#$ header template T((int)(int)|(real)(real))
#$ header function f(T, S)
def f(g, a):
    return g(a)

