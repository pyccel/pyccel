# pylint: disable=missing-function-docstring, missing-module-docstring/


def test_triggered():
    print("Hopefully django is triggered")
    if 5:
        return 2
    else:
        return 6
