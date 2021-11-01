# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.decorators import inline

@inline
def get_powers(s : int):
    return s, s*s, s*s*s

if __name__ == '__main__':
    print(get_powers(3))
    print(get_powers(4))
