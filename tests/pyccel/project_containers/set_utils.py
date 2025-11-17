# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring

def unique_elements(values: list[int]) -> set[int]:
    return set(values)

def common_elements(set1: set[int], set2: set[int]) -> set[int]:
    set3 = set1 & set2
    return set3

