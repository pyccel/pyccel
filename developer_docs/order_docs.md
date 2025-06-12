# ndarrays memory layout (order)

## Order in NumPy

`order` is the parameter given to the `numpy.array` function in order to choose how a multi-dimensional array is stored in memory.
For both of the orders discussed here (`C` and `F`) the arrays are stored **contiguously** in memory, but they differ in how their entries are arranged.

### Order C

`order='C'` tells NumPy to store the array row by row (row-major). For example:

```python
import numpy as np

if __name__ == "__main__":
  a = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]], order='C') # order='C' is the default in numpy.array
  print(a.ravel('K'))
 ```

`array.ravel('k')` shows us how the array is stored in memory.
This Python script will output `[1 2 3 4 5 6 7 8 9]`; notice that the rows are stored one after the other.
This is the default behaviour in Python.

### Order F

`order='F'` on the other hand tells NumPy to store the array column by column (column-major). For example:

```python
import numpy as np

if __name__ == "__main__":
  a = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]], order='F')
  print(a.ravel('K'))
```

This Python script will output `[1 4 7 2 5 8 3 6 9]`; notice that the columns are stored one after the other.

### Printing and indexing in NumPy

The `order` of a NumPy array does not affect the indexing or the printing: unlike the `transpose` operation, the `shape` of the array remains the same, and only the `strides` change. For example:

```python
import numpy as np

if __name__ == "__main__":
   a = np.array([[1, 2],
                [4, 5],
                [7, 8]], order='F')
   b = np.array([[1, 2],
                [4, 5],
                [7, 8]], order='C')
   print(a.shape, a.strides) # output: (3, 2) (8, 24)
   print(b.shape, b.strides) # output: (3, 2) (16, 8)
   print(a)
   # output:[[1 2]
   #         [4 5]
   #         [7 8]]
   print(b)
   # output:[[1 2]
   #         [4 5]
   #         [7 8]]

   print(a[2][1], a[0][0], a[1]) # output: 8 1 [4 5]
   print(b[2][1], b[0][0], b[1]) # output: 8 1 [4 5]
```

`arr.strides` is the variable that helps us navigate the array (indexing/printing) by telling us how many bytes we have to skip in memory to move to the next position along a certain axis (dimension). For example for `memory_layout_of_a = [1 4 7 2 5 8]` and `strides_of_a = (8, 24)`, we have to skip 8 bytes (1 element for `int64`) to move to the next row, but 24 bytes (3 elements for `int64`) to get to the same position in the next column of `a`.
`a[2][1]` would give us `'8'`, using the `strides`: `2 * 8 + 1 * 24 = 40`, which means that in the flattened array, we would have to skip `40` bytes to get the value of `a[2][1]`, each element is 8 bytes, so we would have to skip `40 / 8 = 5` elements to get to `'8'`

The order does however change how the user writes code.
With `order='C'` (as in C), the last dimension contains contiguous elements, whereas with `order='F'` (as in Fortran) the first dimension contains contiguous elements.
Fast code should index efficiently.
By this, we mean that the elements should be visited in the order in which they appear in memory.
For example here is the efficient indexing for 2D arrays:

```python
import numpy as np
if __name__ == "__main__":
   a = np.array([[1, 2],
                [4, 5],
                [7, 8]], order='F')
   b = np.array([[1, 2],
                [4, 5],
                [7, 8]], order='C')
   for row in range(3):
       for col in range(2):
           b[row, col] = ...
   for col in range(2):
       for row in range(3):
           b[row, col] = ...
```

## Pyccel's C code

In Pyccel's C code, we aim to replicate NumPy's indexing/printing and memory layout conventions.

### Ordering in C code

Multidimensional arrays in `C` code are flattened into a one dimensional array, `strides` and `shape` are used to navigate this array (unlike NumPy, Pyccel's strides use 'number of elements' instead of 'number of bytes' as a unit)

### Indexing in C code

For indexing the function `GET_ELEMENT(arr, type, ...)` is used, indexing does not change with `order` so that we can mirror NumPy's conventions.

If we take the following 2D array as an example:

|   |   |   |
|---|---|---|
| 1 | 2 | 3 |
| 4 | 5 | 6 |

with `array.rows = 2` and `array.columns = 3`, `GET_ELEMENT(arr, int32, 0, 1)` which is equivalent to `arr[0][1]` would return `2` no matter the `order`.

To loop efficiently in an `order_c ndarray`, we would do this:

```c
for (int row = 0; row < array.rows; ++i)
{
  for (int column = 0; column < array.columns; ++j)
  {
    (*cspan_at(&x, row, column)) = ...;
  }
}
```

For an `order_f ndarray` we would do this:

```c
for (int column = 0; column < array.columns; ++i)
{
  for (int row = 0; row < array.rows; ++j)
  {
    (*cspan_at(&x, row, column)) = ...;
  }
}
```

## Pyccel's Fortran code

As Fortran has arrays in the language there is no need to add special handling for arrays. Fortran ordered arrays (`order_f`) are already compatible with the Fortran language. They can therefore be passed to the function as they are.

In order to pass C ordered arrays (`order_c`) and retain the shape and correct element placing to be compatible with Fortran, a transpose would be needed.
In Pyccel we prefer to avoid unnecessary copies, so instead we pass the contiguous block of memory to Fortran and change how we index the array to ensure that we access the expected element.

### Ordering in Fortran code

Fortran indexing does not occur in the same order as in C.
If we take the following 2D array as an example:

|   |   |   |
|---|---|---|
| 1 | 2 | 3 |
| 4 | 5 | 6 |

In C the element `A[1,0]=4` is the fourth element in memory, however in Fortran the element `A(1,0)=4` is the second element in memory.
Thus to iterate over this array in the most efficient way in C we would do:

```C
# A.shape = (2,3)
for (int row = 0; row < 2; ++row) {
    for (int column = 0; column < 3; ++column) {
        A[row,column] = ....
    }
}
```

while in Fortran we would do:

```Fortran
# A.shape = (2,3)
do column = 0, 3
    do row = 0, 2
        A(row,column) = ....
    end do
end do
```

As you can see in the Fortran-ordered array the indices are passed to the array in the same order, however the index does not point to the same location in memory.
In C code the index `i_1, i_2, i_3` points to the element `i_1 * (n_2 * n_3) + i_2 * n_2 + i_3` in memory.
In Fortran code the index `i_1, i_2, i_3` points to the element `i_1 + i_2 * n_1 + i_3 * (n_2 * n_3)` in memory.

### Order F

Pyccel's translation of code with `order='F'` should look very similar to the original Python code.

NumPy's storage of the strides ensures that the first dimension is the contiguous dimension as in Fortran, so the code is equivalent for all element-wise operations.

There are some exceptions to this rule, for example printing. Python always prints arrays in a row-major format so Pyccel must take care to respect this rule in the output.

### Order C

As mentioned above, printing a C-ordered array in Fortran is more complicated.
Consider the following 2D C-ordered array:

|   |   |   |
|---|---|---|
| 1 | 2 | 3 |
| 4 | 5 | 6 |

where the numbers indicate the position of the elements in memory. If this data block (`[1, 2, 3, 4, 5, 6]`) were passed to Fortran indicating a size (2,3), we would obtain the following array:

|   |   |   |
|---|---|---|
| 1 | 3 | 5 |
| 2 | 4 | 6 |

As a result we cannot pass the data block without either rearranging the elements (transposing), or changing the index. In Pyccel we prefer avoiding unnecessary copies. As a result we pass a data block (`[1, 2, 3, 4, 5, 6]`), but we indicate a size (3,2). This gives us the following array:

|   |   |
|---|---|
| 1 | 4 |
| 2 | 5 |
| 3 | 6 |

This is equivalent to the transpose of the original array. As a result we can obtain expected results by simply inverting the index order.

Therefore the following Python code

```python
for i in range(2):
    for j in range(3):
        a[i,j] = i*3+j
```

is translated to the following efficient indexing:

```fortran
do i = 0_i64, 1_i64
  do j = 0_i64, 2_i64
    a(j, i) = i * 3_i64 + j
  end do
end do
```

As we are effectively operating on the transpose of the array, this must be taken into account when printing anything related to arrays with `order='C'`.

For example, consider the code:

```python
def f(c_array : 'float[:,:](order=C)', f_array : 'float[:,:](order=F)'):
    print(c_array.shape)
    print(f_array.shape)

    for row in range(c_array.shape[0]):
        for col in range(c_array.shape[1]):
            c_array[row, col] = ...

    for col in range(f_array.shape[1]):
        for row in range(f_array.shape[0]):
            f_array[row, col] = ...
```

This will be translated to:

```Fortran
  subroutine f(c_array, f_array)

    implicit none

    real(f64), intent(inout) :: c_array(0:,0:)
    real(f64), intent(inout) :: f_array(0:,0:)
    integer(i64) :: row
    integer(i64) :: col

    write(stdout, '(A I0 A I0 A)', advance="no") '(' , size(c_array, &
          2_i64, i64) , ', ' , size(c_array, 1_i64, i64) , ')'
    write(stdout, '()', advance="yes")
    write(stdout, '(A I0 A I0 A)', advance="no") '(' , size(f_array, &
          1_i64, i64) , ', ' , size(f_array, 2_i64, i64) , ')'
    write(stdout, '()', advance="yes")
    do row = 0_i64, size(c_array, 2_i64, i64) - 1_i64
      do col = 0_i64, size(c_array, 1_i64, i64) - 1_i64
        c_array(col, row) = ...
      end do
    end do
    do col = 0_i64, size(f_array, 2_i64, i64) - 1_i64
      do row = 0_i64, size(f_array, 1_i64, i64) - 1_i64
        f_array(row, col) = ...
      end do
    end do

  end subroutine f
```

Note the changes to the shape and the indexing, which make this code closer to the following intermediate representation:

```python
def f_intermediate(c_array_T : 'float[:,:](order=F)', f_array : 'float[:,:](order=F)'):
    print(c_array_T.shape[::-1])
    print(f_array.shape)

    for row in range(c_array_T.shape[1]):
        for col in range(c_array_T.shape[0]):
            c_array_T[col, row] = ...

    for col in range(f_array.shape[1]):
        for row in range(f_array.shape[0]):
            f_array[row, col] = ...
```

Note that `f(c_array, f_array) == f_intermediate(c_array.T, f_array)`.
