# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np
from decorators_inline import power_4, get_powers, f, sin_base_1, fill_pi, positron_charge

def g(s : int):
    return f(s)/3

if __name__ == '__main__':
    print(get_powers(5))
    print(power_4(5))
    print(f(4))
    print(sin_base_1(0.25))
    arr = np.empty(4)
    fill_pi(arr)
    print(arr)
    print(positron_charge())
