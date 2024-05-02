# pylint: disable=missing-function-docstring, missing-module-docstring/

__all__ = [
        'one_quote',
        'two_quote',
        'three_quote',
        'concatenate',
        'concatenate_multiple',
        'concatenate_expr',
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
