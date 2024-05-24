# Container types in Pyccel

Pyccel provides support for some container types with certain limits. The types that are currently supported are:
-   NumPy arrays
-   Tuples

## NumPy arrays

NumPy arrays are provided as part of the NumPy support. There is dedicated documentation which explains the limitations and implementation details. See [N-dimensional array](./ndarrays.md) for more details.

## Tuples

In Pyccel tuples are divided into two types: homogeneous and inhomogeneous. Homogeneous tuples are objects where all elements of the container have the same type while inhomogeneous tuples can contain objects of different types. These two types are handled differently and therefore have very different restrictions.

Currently Pyccel cannot wrap tuples so they can be used in functions but cannot yet be exposed to Python.

### Homogeneous tuples

Homogeneous tuples are handled as though they were arrays. This means that they have the same restrictions and advantages as NumPy arrays. In particular they can be indexed at an arbitrary point.

Elements of a homogeneous tuple should have the same type, the same number of dimensions, and (if relevant) the same NumPy ordering. If any of these constraints is not respected then you may unexpectedly find yourself using the more inflexible inhomogeneous tuples. Further tuples containing pointers to other objects cannot always be stored in a homogeneous tuple.

### Inhomogeneous tuples

Inhomogeneous tuples are handled symbolically. This means that an inhomogeneous tuple is treated as a collection of translatable objects. Each of these objects is then handled individually. In particular this means that tuples can only be indexed by compile-time constants.

For example the following code:
```python
def f():
    a = (1, True, 3.0)
    print(a)
    b = a[0]+2
    return a[2]
```
is translated to the following C code:
```c
double f(void)
{
    int64_t a_0;
    bool a_1;
    double a_2;
    int64_t b;
    a_0 = INT64_C(1);
    a_1 = 1;
    a_2 = 3.0;
    printf("%s%"PRId64"%s%s%s%.15lf%s\n", "(", a_0, ", ", a_1 ? "True" : "False", ", ", a_2, ")");
    b = a_0 + INT64_C(2);
    return a_2;
}
```
and the following Fortran code:
```fortran
  function f() result(Out_0001)

    implicit none

    real(f64) :: Out_0001
    integer(i64) :: a_0
    logical(b1) :: a_1
    real(f64) :: a_2
    integer(i64) :: b

    a_0 = 1_i64
    a_1 = .True._b1
    a_2 = 3.0_f64
    write(stdout, '(A, I0, A, A, A, F0.15, A)', advance="no") '(' , a_0 &
          , ', ' , merge("True ", "False", a_1) , ', ' , a_2 , ')'
    write(stdout, '()', advance="yes")
    b = a_0 + 2_i64
    Out_0001 = a_2
    return

  end function f
```

But the following code will raise an error:
```python
def f():
    a = (1, True, 3.0)
    i = 2
    print(a[i])
```
```
ERROR at annotation (semantic) stage
pyccel:
 |fatal [semantic]: foo.py [4,10]| Inhomogeneous tuples must be indexed with constant integers for the type inference to work (a)
```
