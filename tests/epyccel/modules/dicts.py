# pylint: disable=missing-function-docstring, missing-module-docstring


def dict_init():
    a = {1: 1.0, 2: 2.0}
    return a


def dict_empty_init():
    a: "dict[int, float]" = {}
    return a


def pop_element():
    a = {1: 1.0, 2: 2.0}
    return a.pop(1)


def pop_default_element():
    a = {1: True, 2: False}
    return a.pop(3, True)


def pop_bool_keys():
    a = {True: 1, False: 2}
    return a.pop(False)


def pop_falsy_int_default_element():
    a = {1: 2, 2: 3}
    return a.pop(3, 0)


def pop_falsy_bool_default_element():
    a = {1: True, 2: False}
    return a.pop(3, False)


def pop_item():
    a = {1: 1.0, 2: 2.0}
    return a.popitem()


def pop_item_elements():
    a = {1: 1.0, 2: 2.0}
    b = a.popitem()
    return b[0], b[1]


def pop_item_key():
    a = {1: 1.0, 2: 2.0}
    return a.popitem()[0]


def pop_item_expression():
    a = {1: 1.0, 2: 2.0}
    return a.popitem()[0] + 4


def pop_item_unpacking():
    a = {1: 1.0, 2: 2.0}
    b, c = a.popitem()
    return b, c


def getitem_element():
    a = {1: 1.0, 2: 2.0}
    return a[1]


def getitem_modify_element():
    a = {1: 1.0, 2: 2.0}
    a[1] = 3.0
    return a[1]


def dict_contains():
    a = {1: 1.0, 2: 2.0, 3: 3.0}
    return (1 in a), (5 in a), (4.0 in a)


def dict_clear():
    a = {1: 1.0, 2: 2.0}
    a.clear()
    return len(a)


def dict_items():
    a = {1: 1.0, 2: 2.0, 3: 3.0, 5: 4.7}
    key_sum = 0
    val_sum = 0.0
    for key, val in a.items():
        key_sum += key
        val_sum += val

    return key_sum, val_sum


def dict_keys():
    a = {1: 1.0, 2: 2.0, 3: 3.0, 5: 4.7}
    key_sum = 0
    for key in a.keys():  # pylint:disable=consider-iterating-dictionary
        key_sum += key

    return key_sum


def dict_keys_iter():
    a = {1: 1.0, 2: 2.0, 3: 3.0, 5: 4.7}
    key_sum = 0
    for key in a:
        key_sum += key

    return key_sum


def dict_values():
    a = {1: 1.0, 2: 2.0, 3: 3.0, 5: 4.7}
    value_sum = 0.0
    for value in a.values():  # pylint:disable=consider-iterating-dictionary
        value_sum += value

    return value_sum
