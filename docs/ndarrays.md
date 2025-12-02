# The N-dimensional array (ndarray) #

## Description ##

A ndarray is a fixed-size multi-dimensional container of items of the same type and size. The number of dimensions and items in an array is defined by its shape, which is a container of N non-negative integers that specify the sizes of each dimension. The type of items in the array is specified by a separate data-type object, one of which is associated with each ndarray.

Different ndarrays can share the same data, so that changes made in one ndarray may be visible in another. That is, a ndarray can be a "view" to another ndarray, and the data it is referring to is taken care of by the "base" ndarray.
[read more](https://numpy.org/doc/stable/reference/arrays.html)

## Pyccel ndarrays ##

Pyccel relies on [STC](https://github.com/Stclib/STC/) for the implementation of ndarrays in C and uses Fortran's intrinsic arrays for Fortran support. The implementation is very similar to NumPy's ndarrays with some rules due to the difference between the host language (Python) "dynamically typed / internal garbage collector" and the target languages such as C and Fortran which are statically typed languages and don't have a garbage collector.

STC is provided with Pyccel, but any version of this package installed locally will take precedent.

Below we will show some rules that Pyccel has set to handles those differences.

### Dynamically and statically typed languages ###

Generally a variable in Pyccel should always keep its initial type, this also transfers to using the ndarrays.

#### incorrect example ####

```Python
import numpy as np

if __name__ == '__main__':
    a = np.array([1, 2, 3], dtype=float)
    #(some code...)
    a = np.array([1, 2, 3], dtype=int)
```

_OUTPUT_ :

```Shell
ERROR at annotation (semantic) stage
pyccel:
 |error [semantic]: ex.py [5]| Incompatible types in assignment (|a| real <-> int)
```

### Memory management ###

Pyccel makes a difference between ndarrays that own their data and the ones that don't.

Pyccel calls its own garbage collector when needed, but has a set of rules to do so:

-   Can not reassign ndarrays with different ranks.

    ```Python
    import numpy as np

    if __name__ == '__main__':
        a = np.ones((10, 20))
        #(some code...)
        a = np.ones(10)
    ```

    _OUTPUT_ :

    ```Shell
    ERROR at annotation (semantic) stage
    pyccel:
     |error [semantic]: ex.py [4]| Incompatible redefinition (|a| real(10, 20) <-> real(10,))
    ```

This limitation is due to the fact that the rank of Fortran allocatable objects must be specified in their declaration.

-   Can not assign ndarrays that own their data one another.

    ```Python
    import numpy as np

    if __name__ == '__main__':
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([1, 2, 3])
        a = b
    ```

    _OUTPUT_ :

    ```Shell
    ERROR at annotation (semantic) stage
    pyccel:
     |error [semantic]: ex.py [5]| Arrays which own their data cannot become views on other arrays (a)
    ```

    This limitation is due to the fact that the ndarray **a** will have to go from a data owner to a pointer to the **b** ndarray data.

    _NOTE_: this limitation is not applied to assignments which reserve a new memory block.

    -   Python example:

        ```Python
        import numpy as np

        if __name__ == '__main__':
            a = np.ones(20)
            #(some code...)
            a = np.ones(10)
        ```

    -   C equivalent:

        ```C
        #include "ex.h"
        #include <stdint.h>
        #include <stdlib.h>
        #define STC_CSPAN_INDEX_TYPE int64_t
        #include <stc/cspan.h>
        #ifndef _ARRAY_DOUBLE_1D
        #define _ARRAY_DOUBLE_1D
        using_cspan(array_double_1d, double, 1);
        #endif // _ARRAY_DOUBLE_1D

        int main()
        {
            array_double_1d a = {0};
            double* a_ptr;
            double* a_ptr_0001;
            a_ptr = malloc(sizeof(double) * (INT64_C(20)));
            a = (array_double_1d)cspan_md_layout(c_ROWMAJOR, a_ptr, INT64_C(20));
            c_foreach(Dummy_0000, array_double_1d, a) {
                *(Dummy_0000.ref) = 1.0;
            }
            /*(some code...)*/
            free(a.data);
            a.data = NULL;
            a_ptr_0001 = malloc(sizeof(double) * (INT64_C(10)));
            a = (array_double_1d)cspan_md_layout(c_ROWMAJOR, a_ptr_0001, INT64_C(10));
            c_foreach(Dummy_0001, array_double_1d, a) {
                *(Dummy_0001.ref) = 1.0;
            }
            free(a.data);
            a.data = NULL;
            return 0;
        }
        ```

    -   Fortran equivalent:

        ```Fortran
        program prog_prog_ex

          use ex

          use, intrinsic :: ISO_C_Binding, only : f64 => C_DOUBLE , i64 => &
                C_INT64_T

          implicit none

          real(f64), allocatable :: a(:)

          allocate(a(0:19_i64))
          a = 1.0_f64
          !(some code...)
          if (any(size(a) /= [10_i64])) then
            deallocate(a)
            allocate(a(0:9_i64))
          end if
          a = 1.0_f64
          if (allocated(a)) deallocate(a)

        end program prog_prog_ex
        ```

-   Can not reassign to a ndarray that has another pointer accessing its data.

    ```Python
    import numpy as np

    if __name__ == '__main__':
        a = np.ones(10)
        b = a[:5]
        #(some code...)
        a = np.zeros(20)
    ```

    _OUTPUT_ :

    ```Shell
    ERROR at annotation (semantic) stage
    pyccel:
      |error [semantic]: ex.py [6]| Attempt to reallocate an array which is being used by another variable (a)
    ```

This limitation is set since we need to free the previous data when we reallocate the ndarray. In this case, this will cause the data pointed to by the view **b** to became inaccessible.

### Slicing and indexing ###

The indexing and slicing in Pyccel handles only the basic indexing of [numpy arrays](https://numpy.org/doc/stable/user/basics.indexing.html). When multiple indexing expressions are used on the same variable Pyccel squashes them into one object. This means that we do not handle multiple slice indices applied to the same variable (e.g. `a[1::2][2:]`). This is not recommended anyway as it makes code hard to read.

Some examples:

-   Python code:

    ```Python
    import numpy as np

    if __name__ == '__main__':
        a = np.array([1, 3, 4, 5])
        a[0] = 0
    ```

    -   C equivalent:

        ```C
        #include "ex.h"
        #include <stdint.h>
        #include <stdlib.h>
        #define STC_CSPAN_INDEX_TYPE int64_t
        #include <stc/cspan.h>
        #ifndef _ARRAY_INT64_1D
        #define _ARRAY_INT64_1D
        using_cspan(array_int64_1d, int64_t, 1);
        #endif // _ARRAY_INT64_1D

        int main()
        {
            array_int64_1d a = {0};
            int64_t* a_ptr;
            a_ptr = malloc(sizeof(int64_t) * (INT64_C(4)));
            a = (array_int64_1d)cspan_md_layout(c_ROWMAJOR, a_ptr, INT64_C(4));
            (*cspan_at(&a, INT64_C(0))) = INT64_C(1);
            (*cspan_at(&a, INT64_C(1))) = INT64_C(3);
            (*cspan_at(&a, INT64_C(2))) = INT64_C(4);
            (*cspan_at(&a, INT64_C(3))) = INT64_C(5);
            (*cspan_at(&a, INT64_C(0))) = INT64_C(0);
            free(a.data);
            a.data = NULL;
            return 0;
        }
        ```

    -   Fortran equivalent:

        ```Fortran
        program prog_prog_ex

          use ex

          use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T

          implicit none

          integer(i64), allocatable :: a(:)

          allocate(a(0:3_i64))
          a = [1_i64, 3_i64, 4_i64, 5_i64]
          a(0_i64) = 0_i64
          if (allocated(a)) deallocate(a)

        end program prog_prog_ex
        ```

-   Python code:

    ```Python
    import numpy as np

    if __name__ == '__main__':
        a = np.ones((10, 20))
        b = a[2:, :5]
    ```

    -   C equivalent:

        ```C
        #include "ex.h"
        #include <stdint.h>
        #include <stdlib.h>
        #define STC_CSPAN_INDEX_TYPE int64_t
        #include <stc/cspan.h>
        #ifndef _ARRAY_DOUBLE_2D
        #define _ARRAY_DOUBLE_2D
        using_cspan(array_double_2d, double, 2);
        #endif // _ARRAY_DOUBLE_2D

        int main()
        {
            array_double_2d a = {0};
            array_double_2d b = {0};
            double* a_ptr;
            a_ptr = malloc(sizeof(double) * (INT64_C(200)));
            a = (array_double_2d)cspan_md_layout(c_ROWMAJOR, a_ptr, INT64_C(10), INT64_C(20));
            c_foreach(Dummy_0000, array_double_2d, a) {
                *(Dummy_0000.ref) = 1.0;
            }
            b = cspan_slice(array_double_2d, &a, {INT64_C(2), c_END}, {INT64_C(0), INT64_C(5)});
            free(a.data);
            a.data = NULL;
            return 0;
        }
        ```

    -   Fortran equivalent:

        ```Fortran
        program prog_prog_ex

          use ex

          use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T , f64 => &
                C_DOUBLE

          implicit none

          real(f64), allocatable, target :: a(:, :)
          real(f64), pointer :: b(:, :)

          allocate(a(0:19_i64, 0:9_i64))
          a = 1.0_f64
          b(0:, 0:) => a(:4_i64, 2_i64:)
          if (allocated(a)) deallocate(a)

        end program prog_prog_ex
        ```

-   Python code:

    ```Python
    import numpy as np

    if __name__ == '__main__':
        a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        b = a[1]
        c = b[2]
        print(c)
    ```

    -   C equivalent:

        ```C
        #include "ex.h"
        #include <inttypes.h>
        #include <stdint.h>
        #include <stdio.h>
        #include <stdlib.h>
        #define STC_CSPAN_INDEX_TYPE int64_t
        #include <stc/cspan.h>
        #ifndef _ARRAY_INT64_2D
        #define _ARRAY_INT64_2D
        using_cspan(array_int64_2d, int64_t, 2);
        #endif // _ARRAY_INT64_2D

        #ifndef _ARRAY_INT64_1D
        #define _ARRAY_INT64_1D
        using_cspan(array_int64_1d, int64_t, 1);
        #endif // _ARRAY_INT64_1D

        int main()
        {
            array_int64_2d a = {0};
            array_int64_1d b = {0};
            int64_t c;
            int64_t* a_ptr;
            a_ptr = malloc(sizeof(int64_t) * (INT64_C(8)));
            a = (array_int64_2d)cspan_md_layout(c_ROWMAJOR, a_ptr, INT64_C(2), INT64_C(4));
            (*cspan_at(&a, INT64_C(0), INT64_C(0))) = INT64_C(1);
            (*cspan_at(&a, INT64_C(0), INT64_C(1))) = INT64_C(2);
            (*cspan_at(&a, INT64_C(0), INT64_C(2))) = INT64_C(3);
            (*cspan_at(&a, INT64_C(0), INT64_C(3))) = INT64_C(4);
            (*cspan_at(&a, INT64_C(1), INT64_C(0))) = INT64_C(5);
            (*cspan_at(&a, INT64_C(1), INT64_C(1))) = INT64_C(6);
            (*cspan_at(&a, INT64_C(1), INT64_C(2))) = INT64_C(7);
            (*cspan_at(&a, INT64_C(1), INT64_C(3))) = INT64_C(8);
            b = cspan_slice(array_int64_1d, &a, {INT64_C(1)}, {c_ALL});
            c = (*cspan_at(&b, INT64_C(2)));
            printf("%"PRId64"\n", c);
            free(a.data);
            a.data = NULL;
            return 0;
        }
        ```

    -   Fortran equivalent:

        ```Fortran
        program prog_prog_ex

          use ex

          use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T
          use, intrinsic :: ISO_FORTRAN_ENV, only : stdout => output_unit

          implicit none

          integer(i64), allocatable, target :: a(:, :)
          integer(i64), pointer :: b(:)
          integer(i64) :: c

          allocate(a(0:3_i64, 0:1_i64))
          a = reshape([[1_i64, 2_i64, 3_i64, 4_i64], [5_i64, 6_i64, 7_i64, 8_i64 &
                ]], [4_i64, 2_i64])
          b(0:) => a(:, 1_i64)
          c = b(2_i64)
          write(stdout, '(I0)', advance="yes") c
          if (allocated(a)) deallocate(a)

        end program prog_prog_ex
        ```

-   Python code:

    ```Python
    import numpy as np

    if __name__ == '__main__':
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        b = a[1::2][2]
        print(b)
    ```

    -   C equivalent:

        ```C
        #include "ex.h"
        #include <inttypes.h>
        #include <stdint.h>
        #include <stdio.h>
        #include <stdlib.h>
        #define STC_CSPAN_INDEX_TYPE int64_t
        #include <stc/cspan.h>
        #ifndef _ARRAY_INT64_1D
        #define _ARRAY_INT64_1D
        using_cspan(array_int64_1d, int64_t, 1);
        #endif // _ARRAY_INT64_1D

        int main()
        {
            array_int64_1d a = {0};
            int64_t b;
            int64_t* a_ptr;
            a_ptr = malloc(sizeof(int64_t) * (INT64_C(8)));
            a = (array_int64_1d)cspan_md_layout(c_ROWMAJOR, a_ptr, INT64_C(8));
            (*cspan_at(&a, INT64_C(0))) = INT64_C(1);
            (*cspan_at(&a, INT64_C(1))) = INT64_C(2);
            (*cspan_at(&a, INT64_C(2))) = INT64_C(3);
            (*cspan_at(&a, INT64_C(3))) = INT64_C(4);
            (*cspan_at(&a, INT64_C(4))) = INT64_C(5);
            (*cspan_at(&a, INT64_C(5))) = INT64_C(6);
            (*cspan_at(&a, INT64_C(6))) = INT64_C(7);
            (*cspan_at(&a, INT64_C(7))) = INT64_C(8);
            b = (*cspan_at(&a, INT64_C(5)));
            printf("%"PRId64"\n", b);
            free(a.data);
            a.data = NULL;
            return 0;
        }
        ```

    -   Fortran equivalent:

        ```Fortran
        program prog_prog_ex

          use ex

          use, intrinsic :: ISO_C_Binding, only : i64 => C_INT64_T
          use, intrinsic :: ISO_FORTRAN_ENV, only : stdout => output_unit

          implicit none

          integer(i64), allocatable :: a(:)
          integer(i64) :: b

          allocate(a(0:7_i64))
          a = [1_i64, 2_i64, 3_i64, 4_i64, 5_i64, 6_i64, 7_i64, 8_i64]
          b = a(5_i64)
          write(stdout, '(I0)', advance="yes") b
          if (allocated(a)) deallocate(a)

        end program prog_prog_ex
        ```

## NumPy [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) functions/properties progress in Pyccel ##

-   Supported [types](https://numpy.org/devdocs/user/basics.types.html):

    `bool`, `int`, `int8`, `int16`, `int32`, `int64`, `float`, `float32`, `float64`, `complex`, `complex64` and `complex128`. They can be used as cast functions too.

    Note: `np.bool`, `np.int`, `np.float` and `np.complex` are just aliases to the Python native types, and are considered as a deprecated way to work with Python built-in types in NumPy.

-   Properties:

    -   `real`, `imag`, `shape`, `amax`, `amin`

-   Methods:

    -   `sum`
