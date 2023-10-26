# Supported Class by Pyccel

Pyccel strives to provide robust support for object-oriented programming concepts commonly used by developers. In Pyccel, classes are a fundamental building block for creating structured and reusable code. This documentation outlines key features and considerations when working with classes in Pyccel.

:warning: Pyccel's class support is currently limited to translations only. The implementation of the wrapper which will make the class accessible from Python is in progress.

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
    def __init__(self : 'MyClass', param1 : 'int', param2 : 'int'):
        self.param1 = param1
        self.param2 = np.ones(param2)
        print("MyClass Object created!")

    @inline
    def get_param(self : 'MyClass'):
        print(self.param1, self.param2)

class MyClass1:
    def __init__(self : 'MyClass1'):
        print("MyClass1 Object created!")

    def Method1(self : 'MyClass1', param1 : 'MyClass'):
        self.param = param1

    def Method2(self : 'MyClass1'):
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

## Limitations

It's important to note that Pyccel does not support class inheritance, properties, magic methods or class variables. For our first implementation, the focus of Pyccel is primarily on core class functionality and memory management.
