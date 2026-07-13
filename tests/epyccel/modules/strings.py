# pylint: disable=missing-function-docstring, missing-module-docstring

def str_comp():
    a = "hello"
    if a == "world":
        return 1
    elif a != "boo":
        return 2
    elif a == "hello":
        return 3
    else:
        return 4

def str_option_test(option: str):
    if option == "do this":
        return 1.0
    else:
        return 2.0

def string_argument_optional_str_option_test(option: str = None):
    if option is not None and option == "do this":
        return 1.0
    else:
        return 2.0
