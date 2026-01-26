# pylint: disable=missing-function-docstring, missing-module-docstring
from user_cls_mod import A as B

if __name__ == '__main__':
    b = B(3)
    print(b.my_val)
