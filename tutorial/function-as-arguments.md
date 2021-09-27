# Functions as arguments

Note: before reading this you should have read [Installation and Command Line Usage](https://github.com/pyccel/pyccel/blob/master/tutorial/quickstart.md#installation)

Functions as arguments is a regular feature in all the functional languages, this feature helps to write less code by passing directly functions as arguments to other ones, instead of storing the result in variables (which some times thier data types can be too complicated to re-store) then pass them to the target functions. And here is how pyccel converts this feature's code:

this simple example shows the function `print` takes as argument the function `func1` that returns scalar `1`.

this the python code to be translated:

```python
def func1():
    return 1

if __name__ == '__main__':
    print(func1())
```

this is the generated C code by pyccel:

```C
#include "boo.h"
#include <stdint.h>
#include <stdlib.h>


/*........................................*/
int64_t func1(void)
{

    return 1;
}
/*........................................*/
```

This is program C code:

```C
#include "boo.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
int main()
{
    printf("%ld\n", func1());
    return 0;
}
```

And we have also the generated header file:

```C
#ifndef BOO_H
#define BOO_H

#include <stdlib.h>
#include <stdint.h>


int64_t func1(void);
void boo__init(void);
#endif // BOO_H
```

Here is the equivalent fortran code.

This is the module fortran code:

```fortran
module boo


  use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T
  implicit none

  contains

  !........................................
  function func1() result(Out_0001)

    implicit none

    integer(i64) :: Out_0001

    Out_0001 = 1_i64
    return

  end function func1
  !........................................

end module boo
```

And this is the program fortran code:

```fortran
program prog_prog_boo

  use boo

  use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T
  implicit none

  print *, func1()

end program prog_prog_boo
```

in the next example we can see how functions-as-arguments feature can be useful.

Here, the function `print` takes as argument the function `sum` which does the sum of the results of the two functions `num_1` and `num_2`.

This is the python code:

```python
def num_1():
    return 1

def num_2():
    return 2

def sum1(a: 'int', b: 'int'):
	return a + b

if __name__ == '__main__':
    print(sum1(num_1(), num_2()))
```

This is the generated C code:

```C
#include "boo.h"
#include <stdlib.h>
#include <stdint.h>


/*........................................*/
int64_t num_1(void)
{

    return 1;
}
/*........................................*/
/*........................................*/
int64_t num_2(void)
{

    return 2;
}
/*........................................*/
/*........................................*/
int64_t sum1(int64_t a, int64_t b)
{

    return a + b;
}
/*........................................*/
```

we have another generated C code that contains the function `main` to generate a program:

```C
#include "boo.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
int main()
{
    printf("%ld\n", sum1(num_1(), num_2()));
    return 0;
}
```

And this is the header file:

```C
#ifndef BOO_H
#define BOO_H

#include <stdlib.h>
#include <stdint.h>


int64_t num_1(void);
int64_t num_2(void);
int64_t sum1(int64_t a, int64_t b);
#endif // BOO_H
```

Here is the equivalent fortran code.

This is the module fortran code:

```fortran
module boo


  use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T
  implicit none

  contains

  !........................................
  function num_1() result(Out_0001)

    implicit none

    integer(i64) :: Out_0001

    Out_0001 = 1_i64
    return

  end function num_1
  !........................................

  !........................................
  function num_2() result(Out_0002)

    implicit none

    integer(i64) :: Out_0002

    Out_0002 = 2_i64
    return

  end function num_2
  !........................................

  !........................................
  function sum1(a, b) result(Out_0003)

    implicit none

    integer(i64) :: Out_0003
    integer(i64), value :: a
    integer(i64), value :: b

    Out_0003 = a + b
    return

  end function sum1
  !........................................

end module boo
```

this is the program fortran code:

```fortran
program prog_prog_boo

  use boo

  use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T
  implicit none

  print *, sum1(num_1(), num_2())

end program prog_prog_boo
```

## Getting Help

If you face problems with pyccel, please take the following steps:

1.  Consult our documention in the tutorial directory;
2.  Send an email message to pyccel@googlegroups.com;
3.  Open an issue on GitHub.

Thank you!
