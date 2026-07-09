# pylint: disable=missing-function-docstring, missing-module-docstring

def complex_imag():
    a = 1 + 2j
    return a.imag

def complex_imag_expr(a: "complex", b: "complex"):
    return (a + b).imag

def float_imag():
    a = 1.5
    return a.imag

def complex_real():
    a = 1 + 2j
    return a.real

def complex_real_expr(a: "complex", b: "complex"):
    return (a + b).real

def complex_conjugate(a: "complex", b: "complex"):
    return (a + b).conjugate()

def complex64_conjugate(a: "complex64", b: "complex64"):
    return (a + b).conj()

def float_conjugate(a: "float", b: "float"):
    return (a + b).conjugate()

def float64_conjugate(a: "float64", b: "float64"):
    return (a + b).conj()

def int_conjugate(a: "int", b: "int"):
    return (a + b).conjugate()

def int32_conjugate(a: "int32", b: "int32"):
    return (a + b).conj()

def bool_conjugate(a: "bool", b: "bool"):
    return (a or b).conjugate()

def ndarray_var_from_expr(x: "int[:]", y: "int[:]"):
    z = x + y
    a = z.sum()
    return a

def ndarray_var_from_slice(x: "int[:]"):
    z = x[1:]
    a = z.sum()
    return a
