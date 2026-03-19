# pylint: disable=missing-function-docstring, missing-module-docstring
import decorators_inline
import numpy as np
from decorators_inline import (
    f,
    fill_pi,
    get_powers,
    positron_charge,
    power_4,
    sin_base_1,
)


def g(s: int):
    return f(s) / 3


if __name__ == "__main__":
    print(get_powers(5))
    print(power_4(5))
    print(f(4))
    print(sin_base_1(0.25))
    arr = np.empty(4)
    fill_pi(arr)
    print(arr)
    print(positron_charge())

    print(decorators_inline.get_powers(5))
    print(decorators_inline.power_4(5))
    print(decorators_inline.f(4))
    print(decorators_inline.sin_base_1(0.25))
    arr = np.empty(4)
    decorators_inline.fill_pi(arr)
    print(arr)
    print(decorators_inline.positron_charge())
