# pylint: disable=missing-function-docstring, missing-module-docstring

if __name__ == "__main__":
    assert True
    a = 0
    b = a
    assert a == b
    b = 1
    assert a != b
    assert a <= b
    assert b >= a
