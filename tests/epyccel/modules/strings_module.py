# pylint: disable=missing-function-docstring, missing-module-docstring

__all__ = [
        'one_quote',
        'two_quote',
        'three_quote',
        'return_literal',
        'concatenate',
        'concatenate_multiple',
        'concatenate_expr',
        'string_function_call',
        'string_function_call_on_literal',
        'string_function_return',
        ]

def one_quote():
    s = 'hello world'
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
    s = ''
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
    s = 'hello'
    t = str(s)
    return t

def string_function_call_on_literal():
    t = str('hello')
    return t

def string_function_return():
    return str('hello')
