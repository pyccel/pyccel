# `const` keyword

In order to make sure that a function argument is not modified by the function call, Pyccel provides the `const` keyword, which is converted to an equivalent datatype qualifier in the target language. Here is a simple example of its usage:

Here is a working Python code with `const` keyword:

```python
def func1(arr: 'const int[:]'):
    # some code
    return 0
```

The generated C code:

```C
#include "boo.h"
#include "ndarrays.h"
#include <stdint.h>
#include <stdlib.h>


/*........................................*/
int64_t func1(t_ndarray arr)
{
    /*some code*/
    return 0;
}
/*........................................*/
```

The Fortran equivalent:

```fortran
module boo


  use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T
  implicit none

  contains

  !........................................
  function func1(arr) result(Out_0001)

    implicit none

    integer(i64) :: Out_0001
    integer(i64), intent(in) :: arr(0:)

    !some code
    Out_0001 = 0_i64
    return

  end function func1
  !........................................

end module boo
```

Now we will see what happens if we try to modify a constant array:

```Python
def func1(arr: 'const int[:]', i: 'int', v: 'int', z:'int'):
    # some code
    #trying to modify arr
    arr[i] = v*z
    return 0
```

Pyccel will recognise that a constant array cannot be changed and will raise an error similar to:

```sh
ERROR at annotation (semantic) stage
pyccel:
 |fatal [semantic]: boo.py [1]| Cannot modify 'const' argument (arr)
```

## Getting Help

If you face problems with Pyccel, please take the following steps:

1.  Consult our documentation in the tutorial directory;
2.  Send an email message to pyccel@googlegroups.com;
3.  Open an issue on GitHub.

Thank you!
