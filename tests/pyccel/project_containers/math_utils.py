# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring

def square_elements(values: list[int]) -> list[int]:
    out = [v * v for v in values]
    return out

def sum_elements(values: list[int]) -> int:
    out = 0
    for v in values:
        out += v
    return out

