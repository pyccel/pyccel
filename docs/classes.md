# Supported Class by Pyccel

Pyccel strives to provide robust support for object-oriented programming concepts commonly used by developers. In Pyccel, classes are a fundamental building block for creating structured and reusable code. This documentation outlines key features and considerations when working with classes in Pyccel.

## Contents

1. [Constructor Method](#constructor-method)
2. [Destructor Method](#destructor-method)
3. [Class Methods](#class-methods)
4. [Class Properties](#class-properties)
5. [Magic Methods](#magic-methods)
6. [Limitations](#limitations)

## Constructor Method

-   The Constructor Method, `__init__`, is used to initialise the object's attributes.
-   Pyccel only permits one type definition for each of the arguments of the `__init__` method. Union types or templates cannot be used.
-   The first parameter of any method within a class should always be named `self`.

## Destructor Method

-   The Destructor Method, `__del__`, is used to perform cleanup actions when an object is destroyed.

-   Pyccel automatically takes care of garbage collection for classes.
    -   Attributes are released during the destructor's execution.
    -   The class destructor is called automatically once the class goes out of scope.

## Class Methods

-   Pyccel now supports Class Methods and Interfaces.
-   In Pyccel, class attributes can be initialised within any method of the class.
-   Pyccel enables classes to be passed as arguments to methods and functions and returned with modified data.

### - Python Example

```python
import numpy as np
from pyccel.decorators import inline

class MyClass:
    def __init__(self, param1 : 'int', param2 : 'int'):
        self.param1 = param1
        self.param2 = np.ones(param2)
        print("MyClass Object created!")

    @inline
    def get_param(self):
        print(self.param1, self.param2)

class MyClass1:
    def __init__(self):
        print("MyClass1 Object created!")

    def Method1(self, param1 : MyClass):
        self.param = param1

    def Method2(self):
        return MyClass(2, 4)

if __name__ == '__main__':
    obj = MyClass1()
    obj.Method1(obj.Method2())
    obj.param.get_param()
```

### PYTHON _OUTPUT_
```Shell
MyClass1 Object created!
MyClass Object created!
2 [1. 1. 1. 1.]
```

### - Header File Equivalent

```C
struct MyClass {
    int64_t param1;
    t_ndarray param2;
    bool is_freed;
};
struct MyClass1 {
    bool is_freed;
    struct MyClass param;
};

void MyClass__init__(struct MyClass* self, int64_t param1, int64_t param2);
void MyClass__get_param(struct MyClass* self);
void MyClass__del__(struct MyClass* self);
void MyClass1__init__(struct MyClass1* self);
void MyClass1__Method1(struct MyClass1* self, struct MyClass* param1);
struct MyClass MyClass1__Method2(struct MyClass1* self);
void MyClass1__del__(struct MyClass1* self);
```
### - C File Equivalent

```C
/*........................................*/
void MyClass__init__(struct MyClass* self, int64_t param1, int64_t param2)
{
    self->is_freed = 0;
    self->param1 = param1;
    self->param2 = array_create(1, (int64_t[]){param2}, nd_double, false, order_c);
    array_fill((double)1.0, self->param2);
    printf("%s\n", "MyClass Object created!");
}
/*........................................*/
/*........................................*/
void MyClass__del__(struct MyClass* self)
{
    if (!self->is_freed)
    {
        // pass
        free_array(&self->param2);
        self->is_freed = 1;
    }
}
/*........................................*/
/*........................................*/
void MyClass1__init__(struct MyClass1* self)
{
    self->is_freed = 0;
    printf("%s\n", "MyClass1 Object created!");
}
/*........................................*/
/*........................................*/
void MyClass1__Method1(struct MyClass1* self, struct MyClass* param1)
{
    self->param = (*param1);
}
/*........................................*/
/*........................................*/
struct MyClass MyClass1__Method2(struct MyClass1* self)
{
    struct MyClass Out_0001;
    MyClass__init__(&Out_0001, INT64_C(2), INT64_C(4));
    return Out_0001;
}
/*........................................*/
/*........................................*/
void MyClass1__del__(struct MyClass1* self)
{
    if (!self->is_freed)
    {
        // pass
        MyClass__del__(&self->param);
        self->is_freed = 1;
    }
}
/*........................................*/
```

### - C Program File Equivalent
```C
int main()
{
    struct MyClass1 obj;
    struct MyClass Dummy_0000;
    int64_t i;
    MyClass1__init__(&obj);
    Dummy_0000 = MyClass1__Method2(&obj);
    MyClass1__Method1(&obj, &Dummy_0000);
    printf("%"PRId64" ", obj.param.param1);
    printf("%s", "[");
    for (i = INT64_C(0); i < obj.param.param2.shape[INT64_C(0)] - INT64_C(1); i += INT64_C(1))
    {
        printf("%.12lf ", GET_ELEMENT(obj.param.param2, nd_double, (int64_t)i));
    }
    printf("%.12lf]\n", GET_ELEMENT(obj.param.param2, nd_double, (int64_t)obj.param.param2.shape[INT64_C(0)] - INT64_C(1)));
    MyClass1__del__(&obj);
    return 0;
}
```

### C/Valgrind _OUTPUT_

```Shell
MyClass1 Object created!
MyClass Object created!
2 [1.000000000000 1.000000000000 1.000000000000 1.000000000000]

==158858== Memcheck, a memory error detector
==158858== Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.
==158858== Using Valgrind-3.18.1 and LibVEX; rerun with -h for copyright info
==158858== Command: ./MyClass
==158858==
MyClass1 Object created!
MyClass Object created!
2 [1.000000000000 1.000000000000 1.000000000000 1.000000000000]
==158858==
==158858== HEAP SUMMARY:
==158858==     in use at exit: 0 bytes in 0 blocks
==158858==   total heap usage: 4 allocs, 4 frees, 1,072 bytes allocated
==158858==
==158858== All heap blocks were freed -- no leaks are possible
==158858==
==158858== For lists of detected and suppressed errors, rerun with: -s
==158858== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0)
```

## Class Properties

Pyccel now supports class properties (to retrieve a constant value only).

### - Python Example

```python
import numpy as np

class MyClass:
    def __init__(self, param1 : 'int', param2 : 'int'):
        self._param1 = param1
        self._param2 = np.ones(param2)
        print("MyClass Object created!")

    @property
    def param1(self):
        return self._param1

    @property
    def param2(self):
        return self._param2

if __name__ == '__main__':
    obj = MyClass1(2, 4)
    print(obj.param1)
    print(obj.param2)
```

### PYTHON _OUTPUT_
```Shell
MyClass Object created!
2
[1. 1. 1. 1.]
```

### - Header File Equivalent

```C
struct MyClass {
    int64_t private_param1;
    t_ndarray private_param2;
    bool is_freed;
};

void MyClass__init__(struct MyClass* self, int64_t param1, int64_t param2);
int64_t MyClass__param1(struct MyClass* self);
t_ndarray MyClass__param2(struct MyClass* self);
void MyClass__del__(struct MyClass* self);
```

### - C File Equivalent

```C
/*........................................*/
void MyClass__init__(struct MyClass* self, int64_t param1, int64_t param2)
{
    self->is_freed = 0;
    self->private_param1 = param1;
    self->private_param2 = array_create(1, (int64_t[]){param2}, nd_double, false, order_c);
    array_fill((double)1.0, self->private_param2);
    printf("MyClass Object created!\n");
}
/*........................................*/
/*........................................*/
int64_t MyClass__param1(struct MyClass* self)
{
    return self->private_param1;
}
/*........................................*/
/*........................................*/
t_ndarray MyClass__param2(struct MyClass* self)
{
    t_ndarray Out_0001 = {.shape = NULL};
    alias_assign(&Out_0001, self->private_param2);
    return Out_0001;
}
/*........................................*/
/*........................................*/
void MyClass__del__(struct MyClass* self)
{
    if (!self->is_freed)
    {
        // pass
        free_array(&self->private_param2);
        self->is_freed = 1;
    }
}
/*........................................*/
```

### - C Program File Equivalent
```C
int main()
{
    struct MyClass obj;
    t_ndarray Dummy_0000 = {.shape = NULL};
    int64_t i;
    MyClass__init__(&obj, INT64_C(2), INT64_C(4));
    printf("%"PRId64"\n", MyClass__param1(&obj));
    Dummy_0000 = MyClass__param2(&obj);
    printf("[");
    for (i = INT64_C(0); i < Dummy_0000.shape[INT64_C(0)] - INT64_C(1); i += INT64_C(1))
    {
        printf("%.15lf ", GET_ELEMENT(Dummy_0000, nd_double, i));
    }
    printf("%.15lf]\n", GET_ELEMENT(Dummy_0000, nd_double, Dummy_0000.shape[INT64_C(0)] - INT64_C(1)));
    MyClass__del__(&obj);
    free_pointer(&Dummy_0000);
    return 0;
}
```

### - Fortran File Equivalent

```fortran
  type, public :: MyClass
    integer(i64) :: private_param1
    real(f64), allocatable :: private_param2(:)
    logical(b1), private :: is_freed

    contains
    procedure :: create => myclass_create
    procedure :: param1 => myclass_param1
    procedure :: param2 => myclass_param2
    procedure :: free => myclass_free
  end type MyClass

  contains


  !........................................

  subroutine myclass_create(self, param1, param2)

    implicit none

    class(MyClass), intent(inout) :: self
    integer(i64), value :: param1
    integer(i64), value :: param2

    self%is_freed = .False._b1
    self%private_param1 = param1
    allocate(self%private_param2(0:param2 - 1_i64))
    self%private_param2 = 1.0_f64
    write(stdout, '(A)', advance="yes") 'MyClass Object created!'

  end subroutine myclass_create

  !........................................


  !........................................

  function myclass_param1(self) result(Out_0001)

    implicit none

    integer(i64) :: Out_0001
    class(MyClass), intent(inout) :: self

    Out_0001 = self%private_param1
    return

  end function myclass_param1

  !........................................


  !........................................

  subroutine myclass_param2(self, Out_0001)

    implicit none

    real(f64), pointer, intent(out) :: Out_0001(:)
    class(MyClass), target, intent(inout) :: self

    Out_0001(0:) => self%private_param2
    return

  end subroutine myclass_param2

  !........................................


  !........................................

  subroutine myclass_free(self)

    implicit none

    class(MyClass), intent(inout) :: self

    if (.not. self%is_freed) then
      ! pass
      if (allocated(self%private_param2)) then
        deallocate(self%private_param2)
      end if
      self%is_freed = .True._b1
    end if

  end subroutine myclass_free

  !........................................
```

### - Fortran Program File Equivalent

```fortran
  type(MyClass), target :: obj
  real(f64), pointer :: Dummy_0000(:)
  integer(i64) :: i

  call obj % create(2_i64, 4_i64)
  write(stdout, '(I0)', advance="yes") obj % param1()
  call obj % param2(Out_0001 = Dummy_0000)
  write(stdout, '(A)', advance="no") '['
  do i = 0_i64, size(Dummy_0000, kind=i64) - 1_i64 - 1_i64
    write(stdout, '(F0.15, A)', advance="no") Dummy_0000(i) , ' '
  end do
  write(stdout, '(F0.15, A)', advance="no") Dummy_0000(size(Dummy_0000, &
        kind=i64) - 1_i64) , ']'
  write(stdout, '()', advance="yes")
  call obj % free()
```

## Magic Methods

Pyccel supports a subset of magic methods that are listed here:

-   `__add__`
-   `__sub__`
-   `__mul__`
-   `__truediv__`
-   `__pow__`
-   `__lshift__`
-   `__rshift__`
-   `__and__`
-   `__or__`
-   `__iadd__`
-   `__isub__`
-   `__imul__`
-   `__itruediv__`
-   `__ipow__`
-   `__ilshift__`
-   `__irshift__`
-   `__iand__`
-   `__ior__`

Additionally the following methods are supported in the translation but are lacking the wrapper support that would allow them to be called from Python code:

-   `__radd__`
-   `__rsub__`
-   `__rmul__`
-   `__rtruediv__`
-   `__rpow__`
-   `__rlshift__`
-   `__rrshift__`
-   `__rand__`
-   `__ror__`
-   `__contains__`

## Limitations

It's important to note that Pyccel does not support class inheritance, or static class variables. For our first implementation, the focus of Pyccel is primarily on core class functionality and memory management.
