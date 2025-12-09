# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
from math_utils import square_elements, sum_elements
from set_utils import unique_elements, common_elements
from simple_sort import simple_sorted

if __name__ == '__main__':
    data = [1, 2, 3, 2, 4]
    squared = square_elements(data)
    total = sum_elements(squared)

    set1 = [1, 2, 3, 2, 4]
    set2 = {2, 3, 5}

    unique = unique_elements(set1)
    common = common_elements(unique, set2)

    print(squared)
    print(total)
    print(simple_sorted(unique))
    print(simple_sorted(common))

