 # pylint: disable=missing-function-docstring, missing-module-docstring/
import highorder_functions as mod

print(mod.high_int_1(mod.f1, 0))
print(mod.high_int_int_1(mod.f1, mod.f2, 10))
print(mod.high_real_1(mod.f4, 10, 10.5))
print(mod.high_real_2(mod.f7, 999.11, 10.5))
print(mod.high_real_3(mod.f8))
print(mod.high_valuedarg_1(2))
print(mod.high_real_real_int(mod.f7, mod.f4, mod.f3))
print(mod.high_multi_real_1(mod.f9, 44.7, 44.9))
print(mod.high_void_1(mod.f10))
