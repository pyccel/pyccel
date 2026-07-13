# pylint: disable=missing-function-docstring, missing-module-docstring

def f1(x: "int[:]"):
    import numpy

    s = numpy.shape(x)[0]
    return s

def import_from(x: "int[:]"):
    from numpy import shape

    s = shape(x)[0]
    return s

def import_as(x: "int[:]"):
    import numpy as np

    s = np.shape(x)[0]
    return s

def import_method(x: "int[:]"):
    s = x.shape[0]
    return s
