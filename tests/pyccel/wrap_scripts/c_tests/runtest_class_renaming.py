# pylint: disable=missing-module-docstring
from class_renaming import Counter

if __name__ == "__main__":
    c = Counter(10)
    c.increment()
    print(c.value)   # expected 11

