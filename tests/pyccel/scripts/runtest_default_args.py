# pylint: disable=missing-function-docstring, missing-module-docstring
# coding: utf-8
import numpy as np

if __name__ == '__main__':
    from default_args_mod import f1, f5, f3, is_nil_default_arg, recursivity, print_var

    print(f1(2))
    print(f1())

    # ...
    m1 = 3

    x_expected = np.zeros(m1)
    f5(x_expected)
    print(x_expected)
    f5(x_expected, m1)


    print(f3(19.2,6.7))
    print(f3(4.5))
    print(f3(y = 8.2))
    print(f3())

    print(is_nil_default_arg())
    print(is_nil_default_arg(None))
    print(is_nil_default_arg(False))


    print(recursivity(19.2,6.7))
    print(recursivity(4.5))
    print(recursivity(19.2,6.7,True))
    print(recursivity(4.5,z = False))

    print_var()
    print_var(5)
