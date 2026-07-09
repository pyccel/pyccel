# pylint: disable=missing-function-docstring, missing-module-docstring

def modulo_i_i(x: int, y: int):
    return x % y, x % -y, -x % y, -x % -y, y % -y, -y % y

def modulo_r_r(x: "float", y: "float"):
    return x % y, x % -y, -x % y, -x % -y, y % -y, -y % y

def modulo_r_i(x: "float", y: "int"):
    return x % y, x % -y, -x % y, -x % -y, y % -y, -y % y

def modulo_i_r(x: "int", y: "float"):
    return x % y, x % -y, -x % y, -x % -y, y % -y, -y % y

def modulo_multiple(x: "int", y: "float", z: "int"):
    return (
        x % y % z,
        -x % y % z,
        -x % -y % z,
        -x % -y % -z,
        x % -y % z,
        x % -y % -z,
        x % y % -z,
        -x % y % -z,
        -y % y % y,
        y % -y % y,
        y % y % -y,
    )
