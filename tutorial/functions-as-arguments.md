# Functions as arguments

Note: before reading this you should have read [Installation and Command Line Usage](https://github.com/pyccel/pyccel/blob/master/tutorial/quickstart.md#installation)

In order to support passing [function-pointers](https://en.wikipedia.org/wiki/Function_pointer) as arguments. Pyccel needs from the user to define the type of the passed function-pointers, and this can be done by using function-headers. Here is how to use the function header in Pyccel. And how Pyccel converts that feature:

Following the syntax of the function-header which is `#$ header function function_name((func1_return_type)(func1_arguments), (func2_return_type)(func2_arguments), ..., var1_type, var2_type, ...)`. we can pass the correct number and types of arguments (as described in the function-header) to the actual function as shown is the following example:

Note: Pyccel is using the annotated comments, to bring a new syntax to the python code. Those annotated comments will not affect the actual python code but they are necessary for Pyccel to identify the types of the argument, the openMP pragmas, and manny upcoming features.

Here is the Python code:

```Python
#$ header function high_int_int_1((int)(int), (int)(int), int)
def high_int_int_1(function1, function2, a):
    x = function1(a)
    y = function2(a)
    return x + y
```

## Getting Help

If you face problems with pyccel, please take the following steps:

1.  Consult our documention in the tutorial directory;
2.  Send an email message to pyccel@googlegroups.com;
3.  Open an issue on GitHub.

Thank you!
