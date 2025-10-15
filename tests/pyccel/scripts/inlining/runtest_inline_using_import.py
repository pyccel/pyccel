# pylint: disable=missing-function-docstring, missing-module-docstring
import my_other_func as f
from inline_using_import import func_2

if __name__ == '__main__':
    print(f.foobar(3.0))
    print(func_2(1.5))
