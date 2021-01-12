# The N-dimensional array (ndarray)

## Description:
An ndarray is a fixed-size multidimensional container of items of the same type and size. The number of dimensions and items in an array is defined by its shape, which is a container of N non-negative integers that specify the sizes of each dimension. The type of items in the array is specified by a separate data-type object , one of which is associated with each ndarray.

Different ndarrays can share the same data, so that changes made in one ndarray may be visible in another. that is, an ndarray can be a "view" to another ndarray, and the data it is referring to is taken care of by the "base" ndarray.
[read more](https://numpy.org/doc/stable/reference/arrays.html)

## Pyccel ndarrays:
Pyccel use the same implimentation as Numpy ndarrays with some rules due to the diferrences between the host language (Python) "dynamically typed / internal garbage collector" and the target lnguages such as C and Fortran wich they are statically typed languages and don't have a garbage collector.

below we will show some rules that Pyccel has set to handles those diferrences.

### Dynamically and statically typed languages:
a variable in Pyccel should always keep it initial type this also transfer to using the ndarrays.

#### incorrect example:

```Python
import numpy as np

a = np.array([1, 2, 3], dtype=float)
#(some code...)
a = np.array([1, 2, 3], dtype=int)
```
*OUTPUT* :
```
ERROR at annotation (semantic) stage
pyccel:
 |error [semantic]: ex.py [5]| Incompatible types in assignment (|a| real <-> int)
```

### Memory management:
Pyccel make a difference between ndarray that own their data and the ones they don't.
Pyccel call it own garbage collecting when needed but has a set of rules to do so:

1. you can't reassign ndarrays with different ranks.
      ```Python
      import numpy as np

      a = np.ones((10, 20))
      #(some code...)
      a = np.ones(10)
      ```
      *OUTPU* :
      ```
      ERROR at annotation (semantic) stage
      pyccel:
        |error [semantic]: ex.py [4]| Incompatible redefinition (|a| real(10, 20) <-> real(10,))
      ```
      this limitation is the way Fortran alloctables can't change the rank after declaration.
2. you can't assign ndarrays that own their data one another.
      ```Python
      import numpy as np

      a = np.array([1, 2, 3, 4, 5])
      b = np.array([1, 2, 3])
      a = b
      ```
      *OUTPUT* :
      ```
      ERROR at annotation (semantic) stage
      pyccel:
        |error [semantic]: ex.py [5]| Arrays which own their data cannot become views on other arrays (a)
      ```

   this limitation is due to the fact that the ndarray **a** will have to go from a data owner to a pointer to the **b** ndarray data.

   *NOTE*: this limitation does not include reassinging with a new data with respecting the previous rule.
    ```Python
    import numpy as np

    a = np.ones(20)
    #(some code...)
    a = np.ones(10)
    ```
    this will be translated to the following code:
    - in C:
    ```C
    #include <ndarrays.h>
    #include <stdlib.h>
    int main()
    {
        t_ndarray a;

        a = array_create(1, (int32_t[]){20}, nd_double);
        array_fill((double)1.0, a);
        /*(some code...)*/
        if (a.shape != NULL)
        {
            free_array(a); //this line handles the redefinition of the array
        }
        a = array_create(1, (int32_t[]){10}, nd_double);
        array_fill((double)1.0, a);
        free_array(a); //garbage collection at the end of the program
        return 0;
    }
    ```

    - in Fortran:
    ```Fortran
    program prog_ex

    use, intrinsic :: ISO_C_BINDING

    implicit none

    real(C_DOUBLE), allocatable :: a(:)

    allocate(a(0:20_C_INT64_T - 1_C_INT64_T))
    a = 1.0_C_DOUBLE
    !(some code...)
    if (allocated(a)) then
      if (any(size(a) /= [10_C_INT64_T])) then
        deallocate(a)
        allocate(a(0:10_C_INT64_T - 1_C_INT64_T))
      end if
    else
      allocate(a(0:10_C_INT64_T - 1_C_INT64_T))
    end if
    a = 1.0_C_DOUBLE

    end program prog_ex
    ```
3. you can't reassign to an ndarray that has another ndarray acessing his data.

   ```Python
   import numpy as np

   a = np.ones(10)
   b = a[:5]
   #(some code...)
   a = np.zeros(20)
   ```
   *OUTPUT* :
    ```
    ERROR at annotation (semantic) stage
    pyccel:
     |error [semantic]: ex.py [6]| Attempt to reallocate an array which is being used by another variable (a)
    ```
    this limitations is set due to the fact that we need to free the previous data when trying to reallocate an ndarray which in this case will cause the data that the **b** view point to the becaume inacessible.

### Slicing and indexing.

the indexing and slicing in Pyccel handles only the basic indexing of [numpy arrays](https://numpy.org/doc/stable/user/basics.indexing.html).

Some examples:

1. Python code:
```Python
import numpy as np

a = np.array([1, 3, 4, 5])
a[0] = 0
```

- C equivalent:
```C
#include <ndarrays.h>
#include <stdint.h>
#include <stdlib.h>
int main()
{
    t_ndarray a;



    a = array_create(1, (int32_t[]){4}, nd_int64);
    int64_t array_dummy_0001[] = {1, 3, 4, 5};
    memcpy(a.nd_int64, array_dummy_0001, a.buffer_size);

    a.nd_int64[get_index(a, 0)] = 0;
    free_array(a);
    return 0;
}
```

- Fortran equivalent:
```Fortran
program prog_ex

use, intrinsic :: ISO_C_BINDING

implicit none

integer(C_INT64_T), allocatable :: a(:)

allocate(a(0:4_C_INT64_T - 1_C_INT64_T))
a = [1_C_INT64_T, 3_C_INT64_T, 4_C_INT64_T, 5_C_INT64_T]
a(0_C_INT64_T) = 0_C_INT64_T

end program prog_ex
```

2. Python code:
```Python
import numpy as np

a = np.ones((10, 20))
b = a[2:, :5]
```

- C equivalent:
```C
#include <stdlib.h>
#include <ndarrays.h>
int main()
{
    t_ndarray a;
    t_ndarray b;



    a = array_create(2, (int32_t[]){10, 20}, nd_double);
    array_fill((double)1.0, a);

    b = array_slicing(a, new_slice(2, a.shape[0], 1), new_slice(5, a.shape[1], 1));
    free_array(a);
    free_pointer(b);
    return 0;
}
```

- Fortran equivalent:
```Fortran
program prog_ex

use, intrinsic :: ISO_C_BINDING

implicit none

real(C_DOUBLE), allocatable, target :: a(:,:)
real(C_DOUBLE), pointer :: b(:,:)

allocate(a(0:20_C_INT64_T - 1_C_INT64_T, 0:10_C_INT64_T - 1_C_INT64_T))
a = 1.0_C_DOUBLE
b(0:, 0:) => a(5_C_INT64_T:, 2_C_INT64_T:)

end program prog_ex
```

3. Python code:
```Python
import numpy as np

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
b = a[1]
c = b[2]
print(c)
```

- C equivalent:
```C
#include <stdint.h>
#include <stdlib.h>
#include <ndarrays.h>
#include <stdio.h>
int main()
{
    t_ndarray a;
    t_ndarray b;
    int64_t c;



    a = array_create(2, (int32_t[]){2, 4}, nd_int64);
    int64_t array_dummy_0001[] = {1, 2, 3, 4, 5, 6, 7, 8};
    memcpy(a.nd_int64, array_dummy_0001, a.buffer_size);

    b = array_slicing(a, 1, new_slice(1, 1+1, 1), new_slice(0, a.shape[1], 1));
    c = b.nd_int64[get_index(b, 2)];
    printf("%ld\n", c);
    free_array(a);
    free_pointer(b);
    return 0;
}
```

- Fortran equivalent:
```Fortran
program prog_ex

use, intrinsic :: ISO_C_BINDING

implicit none

integer(C_INT64_T), allocatable, target :: a(:,:)
integer(C_INT64_T), pointer :: b(:)
integer(C_INT64_T) :: c

allocate(a(0:4_C_INT64_T - 1_C_INT64_T, 0:2_C_INT64_T - 1_C_INT64_T))
a = reshape([[1_C_INT64_T, 2_C_INT64_T, 3_C_INT64_T, 4_C_INT64_T], [ &
      5_C_INT64_T, 6_C_INT64_T, 7_C_INT64_T, 8_C_INT64_T]], [ &
      4_C_INT64_T, 2_C_INT64_T])
b(0:) => a(:, 1_C_INT64_T)
c = b(2_C_INT64_T)
print *, c

end program prog_ex
```
