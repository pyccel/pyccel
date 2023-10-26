# Supported Class by Pyccel

Pyccel strives to provide robust support for object-oriented programming concepts commonly used by developers. In Pyccel, classes are a fundamental building block for creating structured and reusable code. This documentation outlines key features and considerations when working with classes in Pyccel.

## Constructor Method

-   The Constructor Method, `__init__`, is used to initialise the object's attributes.
-   It must be named `__init__`, or the Pyccel compiler will generate an error.
-   The first parameter of the `Constructor Method` should always be named `self`.
-   Pyccel only permits one type definition for each of the arguments of the `__init__` method. Union types or templates cannot be used.

## Destructor Method

-   The Destructor Method, `__del__`, is used to perform cleanup actions when an object is destroyed.
-   Pyccel automatically takes care of garbage collection for classes.
    - Attributes are released during the destructor's execution.
    - The class destructor is called automatically once the class goes out of scope.
-   The first parameter of the `Destructor Method` should always be named `self`.

### - Python Example

```python
import numpy as np
class MyClass:
    def __init__(self : 'MyClass', param1 : 'int', param2 : 'float[:]'):
        self.param1 = param1
        self.param2 = param2
        print("Object created!")
        print(param1, param2)

if __name__ == "__main__":
    obj = MyClass(1, np.ones(4))
```

### PYTHON _OUTPUT_
```Shell
Object created!
1 [1. 1. 1. 1.]
```

### - Header File Equivalent

```C
#ifndef SBOOF_H
#define SBOOF_H

struct MyClass {
    int64_t param1;
    t_ndarray param2;
    bool is_freed;
};

void MyClass__init__(struct MyClass* self, int64_t param1, t_ndarray param2);
void MyClass__del__(struct MyClass* self);

#endif // sboof_H
```
### - C File Equivalent

```C
/*........................................*/
void MyClass__init__(struct MyClass* self, int64_t param1, t_ndarray param2)
{
    self->is_freed = 0;
    self->param1 = param1;
    alias_assign(&self->param2, param2);
    printf("%s\n", "Object created!");
}
/*........................................*/
/*........................................*/
void MyClass__del__(struct MyClass* self)
{
    if (!self->is_freed)
    {
        // pass
        free_pointer(&self->param2);
        self->is_freed = 1;
    }
}
/*........................................*/
```

### - C Program File Equivalent
```C
int main()
{
    t_ndarray Dummy_0000 = {.shape = NULL};
    struct MyClass obj;
    Dummy_0000 = array_create(1, (int64_t[]){INT64_C(1)}, nd_double, false, order_c);
    array_fill((double)1.0, Dummy_0000);
    MyClass__init__(&obj, INT64_C(1), Dummy_0000);
    free_array(&Dummy_0000);
    MyClass__del__(&obj);
    return 0;
}
```

### C/Valgrind _OUTPUT_

```Shell
Object created!
1 [1.000000000000]

==151606== Memcheck, a memory error detector
==151606== Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.
==151606== Using Valgrind-3.18.1 and LibVEX; rerun with -h for copyright info
==151606== Command: ./MyClass
==151606==
Object created!
1 [1.000000000000]
==151606==
==151606== HEAP SUMMARY:
==151606==     in use at exit: 0 bytes in 0 blocks
==151606==   total heap usage: 6 allocs, 6 frees, 1,064 bytes allocated
==151606==
==151606== All heap blocks were freed -- no leaks are possible
==151606==
==151606== For lists of detected and suppressed errors, rerun with: -s
==151606== ERROR SUMMARY: 0 errors from 0 contexts (suppressed: 0 from 0
```

## Class Methods

-   Pyccel now supports Class Methods and Interfaces.
-   In Pyccel, class attributes can be initialised within any method of the class.
-   Pyccel enables classes to be passed as arguments to methods and functions and returned with modified data.
-   The first parameter of the Class Method should always be named `self`.

### - Python Example

```Python
class MyClass:
    def __init__(self : 'MyClass', param1 : 'int'):
        self.param1 = param1
        print("MyClass Object created!")

class MyClass1:
    def __init__(self : 'MyClass1'):
        print("MyClass1 Object created!")

    def Method1(self : 'MyClass1', param1 : 'MyClass'):
        self.param = param1

    def Method2(self : 'MyClass1'):
        return MyClass(1)

if __name__ == '__main__':
    obj = MyClass1()
    obj.Method1(obj.Method2())
    print(obj.param.param1)
```
### - PYTHON _OUTPUT_

```Shell
MyClass1 Object created!
MyClass Object created!
1
```

### - C Header File Equivalent

```C
#ifndef SBOOF_H
#define SBOOF_H

struct MyClass {
    int64_t param1;
    bool is_freed;
};
struct MyClass1 {
    bool is_freed;
    struct MyClass param;
};

void MyClass__init__(struct MyClass* self, int64_t param1);
void MyClass__del__(struct MyClass* self);
void MyClass1__init__(struct MyClass1* self);
void MyClass1__Method1(struct MyClass1* self, struct MyClass* param1);
struct MyClass MyClass1__Method2(struct MyClass1* self);
void MyClass1__del__(struct MyClass1* self);

#endif // sboof_H
```
### - C File Equivalent

```C
/*........................................*/
void MyClass__init__(struct MyClass* self, int64_t param1)
{
    self->is_freed = 0;
    self->param1 = param1;
    printf("%s\n", "MyClass Object created!");
}
/*........................................*/
/*........................................*/
void MyClass__del__(struct MyClass* self)
{
    if (!self->is_freed)
    {
        // pass
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
    MyClass__init__(&Out_0001, INT64_C(1));
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
    MyClass1__init__(&obj);
    Dummy_0000 = MyClass1__Method2(&obj);
    MyClass1__Method1(&obj, &Dummy_0000);
    printf("%"PRId64"\n", obj.param.param1);
    MyClass1__del__(&obj);
    return 0;
}
```

### - C _OUTPUT_

```Shell
MyClass1 Object created!
MyClass Object created!
1
```

## Limitations

It's important to note that Pyccel does not support class wrappers, inheritance, properties, magic methods, class variables, or inline class methods. The focus of Pyccel is primarily on core class functionality and memory management.
