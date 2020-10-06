# pylint: disable=missing-function-docstring, missing-module-docstring/
{a: 2, 'b':4}
~a

#$ this is invalid comment

for cls in [B, C]:
    try:
        raise cls()
    except B:
        print("B")
    except C:
        print("C")
    finally:
        print("executing finally clause")
