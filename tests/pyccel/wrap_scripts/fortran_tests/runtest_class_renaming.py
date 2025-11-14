# pylint: disable=missing-module-docstring,missing-function-docstring
from typing import Final
from class_renaming import Counter

def print_val(c : Final[Counter]):
    print(c.value)

if __name__ == "__main__":
    c = Counter(10)
    c.increment()
    c.increment()
    print_val(c)
