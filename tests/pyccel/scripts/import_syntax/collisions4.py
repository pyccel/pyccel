# pylint: disable=missing-function-docstring, missing-module-docstring/
import user_mod
import user_mod2

test = user_mod.user_func(1.,2.,3.) + user_mod2.user_func(4.,5.)

print(test)
