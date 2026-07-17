# pylint: disable=missing-function-docstring, missing-module-docstring

__all__ = [
    "one_quote",
    "two_quote",
    "three_quote",
    "return_literal",
    "concatenate",
    "concatenate_multiple",
    "concatenate_expr",
    "string_function_call",
    "string_function_call_on_literal",
    "string_function_return",
]


def one_quote():
    s = "hello world"
    return s


def two_quote():
    s = "hello world"
    return s


def three_quote():
    s = """hello world"""
    return s


def return_literal():
    return "hello world"


def empty_string():
    s = ""
    return s


def concatenate():
    s = "hello"
    t = " world"
    v = s + t
    return v


def concatenate_multiple():
    s = "hello"
    t = "world"
    l = "_"
    v = s + l + t
    return v


def concatenate_expr():
    s = "hello"
    t = "world"
    v = s + "_" + t
    return v


def string_function_call():
    s = "hello"
    t = str(s)
    return t


def string_function_call_on_literal():
    t = str("hello")
    return t


def string_function_return():
    return str("hello")


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
