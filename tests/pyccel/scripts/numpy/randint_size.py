# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np


def create_randint_array(n: "int", high: "int"):
    return np.random.randint(high, size=n)


if __name__ == "__main__":
    x = create_randint_array(10, 100)
    print(len(x))
