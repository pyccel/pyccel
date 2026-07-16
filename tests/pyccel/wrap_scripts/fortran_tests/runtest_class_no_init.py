# pylint: disable=missing-module-docstring
from class_no_init import MethodCheck

if __name__ == "__main__":
    m = MethodCheck()

    m.stash_value(42)
    print(m.my_value)
