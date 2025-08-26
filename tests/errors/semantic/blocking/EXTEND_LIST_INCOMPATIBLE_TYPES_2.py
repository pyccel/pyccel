# Array definition in for loop may cause memory reallocation at each cycle. Consider creating the array before the loop
# pylint: disable=missing-function-docstring, missing-module-docstring

a = [1,2,3]
b = (((4,5), (3, 4)))
a.extend(b)
print(a)
