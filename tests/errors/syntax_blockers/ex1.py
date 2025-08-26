# Annotated comments must start with omp, acc or header
# pylint: disable=missing-function-docstring, missing-module-docstring
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
