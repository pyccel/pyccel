# pylint: disable=missing-function-docstring, missing-module-docstring

#$ header function ln_python(double)
def ln_python (X) :
    return (X-1) - (X-1)**2 / 2 + (X-1)**3 / 3 - (X-1)**4 / 4 + (X-1)**5 / 5 - (X-1)**6 / 6 + (X-1)**7 / 7 - (X-1)**8 / 8 + (X-1)**9 / 9

#$ header function ln_python_exp(double)
def ln_python_exp (Y) :
    x = (Y - 1)
    x2 = x*x
    x4 = x2*x2
    x6 = x4*x2
    x8 = x4*x4
    return x - x2 / 2 + x * x2 / 3 - x4 / 4 + x * x4 / 5 - x6 / 6 + x6 * x / 7 - x8 / 8 + x8 * x / 9

if __name__ == '__main__':
    print(ln_python(2.0))
    print(ln_python_exp(2.0))
