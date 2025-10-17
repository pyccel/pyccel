# Container types in Pyccel

Pyccel provides support for some container types with certain limits. The types that are currently supported are:

-   NumPy arrays
-   Tuples
-   Lists
-   Sets
-   Dictionaries

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
double mod__f(void)
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

```none
ERROR at annotation (semantic) stage
pyccel:
 |fatal [semantic]: foo.py [4,10]| Inhomogeneous tuples must be indexed with constant integers for the type inference to work (a)
```

## Lists/Sets/Dictionaries

Homogeneous lists, sets and dictionaries are implemented using external libraries. In C we rely on [STC](https://github.com/stclib/STC). In Fortran we rely on [gFTL](https://github.com/goddard-Fortran-Ecosystem/gFTL/).

For example the following code:

```python
def f():
    my_list = [1, 2, 3, 4]
    my_set = {1, 2, 3, 4}
    my_dict = {1:1.0, 2:2.0}
    b = my_list[0]+2
    return b + my_set.pop() + my_dict[1]
```

is translated to the following C code:

```c
double mod__f(void)
{
    vec_int64_t my_list = {0};
    hset_int64_t my_set = {0};
    hmap_int64_t_double my_dict = {0};
    int64_t b;
    double result;
    my_list = c_make(vec_int64_t, {INT64_C(1),INT64_C(2),INT64_C(3),INT64_C(4)});
    my_set = c_make(hset_int64_t, {INT64_C(1),INT64_C(2),INT64_C(3),INT64_C(4)});
    my_dict = c_make(hmap_int64_t_double, {{INT64_C(1), 1.0}, {INT64_C(2), 2.0}});
    b = (*vec_int64_t_at(&my_list, INT64_C(0))) + INT64_C(2);
    result = b + hset_int64_t_pop(&my_set) + (*hmap_int64_t_double_at(&my_dict, INT64_C(1)));
    hset_int64_t_drop(&my_set);
    vec_int64_t_drop(&my_list);
    hmap_int64_t_double_drop(&my_dict);
    return result;
}
```

and the following Fortran code:

```fortran
  function f() result(result_0001)

    implicit none

    real(f64) :: result_0001
    type(Vector_integer8) :: my_list
    type(Set_integer8) :: my_set
    type(Map_integer8__real8) :: my_dict
    integer(i64) :: b

    my_list = Vector_integer8([1_i64, 2_i64, 3_i64, 4_i64])
    my_set = Set_integer8([1_i64, 2_i64, 3_i64, 4_i64])
    my_dict = Map_integer8__real8([Pair_integer8__real8(1_i64, 1.0_f64), &
          Pair_integer8__real8(2_i64, 2.0_f64)])
    b = my_list%of(1_i64) + 2_i64
    result_0001 = b + Set_integer8_pop(my_set) + my_dict % of( 1_i64 )
    return

  end function f
```

### Lists of lists and more

Containers such as lists, sets and dictionaries can also contain other containers. In this case memory management is critical to ensure that the memory is shared as it would be in Python. Consider the following example:

```python
def f():
    a = [1, 2, 3]
    b = [a, [4, 5, 6]]
    c = b[1]
    a[0] = 4 # This modifies b
    c[0] = 7 # This modifies b
```

The memory deallocation is not trivial in this case. As a result managed memory counting is used.
The example above is translated to the following C code:

```c
void mod__f(void)
{
    vec_vec_int64_t_mem b = {0};
    vec_int64_t_mem a_mem = vec_int64_t_mem_make(vec_int64_t_init());
    vec_int64_t_mem c_mem;
    (*a_mem.get) = c_make(vec_int64_t, {INT64_C(1),INT64_C(2),INT64_C(3)});
    b = c_make(vec_vec_int64_t_mem, {
        vec_int64_t_mem_clone(a_mem),
        vec_int64_t_mem_make(c_make(vec_int64_t, {INT64_C(4),INT64_C(5),INT64_C(6)}))
    });
    c_mem = vec_int64_t_mem_clone(*vec_vec_int64_t_mem_at(&b, INT64_C(1)));
    (*vec_int64_t_at_mut(a_mem.get, INT64_C(0))) = INT64_C(4);
    (*vec_int64_t_at_mut(c_mem.get, INT64_C(0))) = INT64_C(7);
    vec_vec_int64_t_mem_drop(&b);
    vec_int64_t_mem_drop(&a_mem);
    vec_int64_t_mem_drop(&c_mem);
}
```
