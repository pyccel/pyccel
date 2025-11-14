# pylint: disable=missing-module-docstring,missing-function-docstring
from typing import Final
from class_renaming import Counter

def foo():
    c = Counter(10)
    c.increment()
    print(c.value)   # expected 11

def bar(c : Final[Counter]):
    print(c.value)

if __name__ == "__main__":
    foo()
    c = Counter(10)
    c.increment()
    c.increment()
    bar(c)
