# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring

def simple_sorted(values: set[int]) -> list[int]:
    sorted_list: list[int] = []
    for val in values:
        inserted = False
        for i, l_val in enumerate(sorted_list):
            if val < l_val:
                sorted_list.insert(i, val)
                inserted = True
                break
        if not inserted:
            sorted_list.append(val)
    return sorted_list

