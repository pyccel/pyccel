# pylint: disable=missing-function-docstring, missing-module-docstring
import user_mod
import user_mod2

if __name__ == "__main__":
    test = user_mod.user_func(1.0, 2.0, 3.0) + user_mod2.user_func(4.0, 5.0)

    print(test)
