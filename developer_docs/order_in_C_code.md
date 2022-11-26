# Multidimensional ndarrays memory layout (order)

## Order in numpy

`order` is the parameter given to the `numpy.array` function in order to choose how the array is stored in memory.  
Both of the orders discussed here (`order_f` and `order_c`) are stored **contiguously** in memory, but they differ in how they are arranged.

### Order F

`order='F'` tells `Numpy` to store the array column by column (column-major), example:  

```python
import numpy as np

if __name__ == "__main__":
  a = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]], order='F')
  print(a.ravel('K'))
```  

`array.ravel('k')` shows us how the array is stored in memroy, this python script will output `[1 4 7 2 5 8 3 6 9]`, notice that the columns are stored one after the other.  

### Order C

`order='C'` on the other hand tells `Numpy` to store the array row by row (row-major), example:  

```python
import numpy as np

if __name__ == "__main__":
  a = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]], order='C') # order='C' is the default in numpy.array
  print(a.ravel('K'))
 ```  
 
This python script will output `[1 2 3 4 5 6 7 8 9]`, notice that the rows are stored one after the other.

### Printing and indexing in `Numpy`

`order` in `Numpy` does not affect the indexing or the printing, unlike `transposing`, the `shape` of the array remains the same, only the `strides` change, example:  

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

`arr.strides` is the variable that helps us navigate the array (indexing/printing), it tells us how many bytes we have to skip in memory to move to the next position along a certain axis (dimension). For example for `memory_layout_of_a = [1 4 7 2 5 8]` and `strides_of_a = (8, 24)`, we have to skip 8 bytes (1 element for `int64`) to move to the next row, but 24 bytes (3 elements for `int64`) to get to the same position in the next column of `a`.  
`a[2][1]` would give us `'8'`, using the `strides`: `2 * 8 + 1 * 24 = 40`, which means that in the flattened array, we would have to skip `40` bytes to get the value of `a[2][1]`, each element is 8 bytes, so we would have to skip `40 / 8 = 5` elements to get to `'8'`

## Pyccel's C code

In `Pyccel`'s C code, we try to clone `Numpy`'s indexing/priting and memory layout conventions.

### Ordering in C code

The arrays in `C` code, are flattened into a one dimensional string, `strides` and `shape` are used to navigate the array (unlike `Numpy`. `Pyccel`'s strides use 'number of elements' instead of 'number of bytes' as a unit)
While the `order_c ndarrays` only require a simple copy to be populated, `order_f` array creation requires slightly different steps.  

Example:  

  `order_c` creation  
   &nbsp;&nbsp;&nbsp;&nbsp;1. allocate/create `order_c ndarray`  
   &nbsp;&nbsp;&nbsp;&nbsp;2. copy values to `ndarray`  

  `order_f` creation  
   &nbsp;&nbsp;&nbsp;&nbsp;1. allocate/create temporary `order_c ndarray`  
   &nbsp;&nbsp;&nbsp;&nbsp;2. copy values to temporary `ndarray`  
   &nbsp;&nbsp;&nbsp;&nbsp;3. allocate/create `order_f ndarray`  
   &nbsp;&nbsp;&nbsp;&nbsp;4. copy temporary `ndarray` elements to final `order_f ndarray` using `strides` and `shape`, this will create a column-major version of the temporary `order_c ndarray`  

### Indexing in C code

For indexing, the function `GET_ELEMENT(arr, type, ...)` is used, indexing does not change with `order` so that we can  mirror `Numpy`'s conventions.

If we take the following 2D array as an example:
|   |   |   |
|---|---|---|
| 1 | 2 | 3 |
| 4 | 5 | 6 |

with `array.rows = 2` and `array.columns = 3`, `GET_ELEMENT(arr, int32, 0, 1)` which is equivelant to `arr[0][1]` would return `2` no matter the `order`.  

To loop efficiently in an (`order_c`) array, we would do this:  
```c
for (int i = 0; i < array.rows; ++i)
{
  for (int j = 0; j < array.columns; ++j)
  {
    GET_ELEMENT(array, int32, i, j) = ...;
  }
}
```

For (`order_f`) we would do this:

```c
for (int i = 0; i < array.columns; ++i)
{
  for (int j = 0; j < array.rows; ++j)
  {
    GET_ELEMENT(array, int32, j, i) = ...;
  }
}
```

## `order_c` array creation example

To create an `order_c ndarray`, we simply copy the flattened data to our `ndarray`'s data placeholder that changes depending on the type.  

If the data is composed of scalars only (ex: `np.array([1, 2, 3])`), an `array_dummy` is created, before copying it to our destination `ndarray`.  

Example:  

```python
if __name__ == "__main__":
  import numpy as np
  a = np.array([[1, 2, 3], [4, 5, 6]])
```  

Would translate to:

```c
int main()
{
    t_ndarray a = {.shape = NULL};
    a = array_create(2, (int64_t[]){INT64_C(2), INT64_C(3)}, nd_int64, false, order_c);
    int64_t array_dummy[] = {INT64_C(1), INT64_C(2), INT64_C(3), INT64_C(4), INT64_C(5), INT64_C(6)}; // Creation of an array_dummy containing the scalars, notice the data is flattened
    memcpy(a.nd_int64, array_dummy, 6 * a.type_size); // Copying from array_dummy to our ndarray 'a'
    free_array(a);
    return 0;
}
```  
  
If the data is composed of at least one variable array (like `c` in the example below), we would use a series of copy operations to our `
ndarray`.  

Example:  

```python
if __name__ == "__main__":
  import numpy as np
  a = np.array([1, 2, 3])
  b = np.array([4, 5, 6])
  c = np.array([a, [7, 8, 9], b])
```  

Would translate to this:  

```c
int main()
{
    t_ndarray a = {.shape = NULL};
    t_ndarray b = {.shape = NULL};
    t_ndarray c = {.shape = NULL};
    a = array_create(1, (int64_t[]){INT64_C(3)}, nd_int64, false, order_c);
    int64_t array_dummy[] = {INT64_C(1), INT64_C(2), INT64_C(3)};
    memcpy(a.nd_int64, array_dummy, 3 * a.type_size);
    b = array_create(1, (int64_t[]){INT64_C(3)}, nd_int64, false, order_c);
    int64_t array_dummy_0001[] = {INT64_C(4), INT64_C(5), INT64_C(6)};
    memcpy(b.nd_int64, array_dummy_0001, 3 * b.type_size);
    
    // 'c' ndarray creation starts here, 'c' is [a, [7, 8, 9], b]
    
    c = array_create(2, (int64_t[]){INT64_C(3), INT64_C(3)}, nd_int64, false, order_c); // Allocating 'c' ndarray
    uint32_t offset = 0; // Initializing offset, used later to avoid overwritting data when executing multiple copy operations
    array_copy_data(&c, a, offset); // Copying the first element of 'c', 'offset' is 0 since it's our first copy operation
    offset += a.length; // Incrementing offset for upcoming copy operation
    int64_t array_dummy_0002[] = {INT64_C(7), INT64_C(8), INT64_C(9)}; // Creating an array_dummy with 'c''s second element's scalars ([7, 8, 9])
    memcpy(c.nd_int64 + offset, array_dummy_0002, 3 * c.type_size); // 'offset' is also with 'memcpy'
    offset += 3; // incrementing 'offset', preparing for final copy
    array_copy_data(&c, b, offset); // Copying the third element to 'c' ndarray
    free_array(a);
    free_array(b);
    free_array(c);
    return 0;
}
```  

## `order_f` array creation example

If the data is one dimensional, all we would need is one copy operation, same as an `order_c ndarray`. // TODO: change to this

For (`order_f`), the process is similar to (`order_c`), but instead of copying our data straight to the destination `ndarray`, we first create an (`order_c`) `temp_ndarray`, copy the data to the `temp_ndarray`, then create an (`order_f`) `ndarray`, and copy from the `temp_ndarray` to the destination (`order_f`) `ndarray` _ using `strides` and `shape` _ to get the correct column-major memory layout.  

Example:  

```python
if __name__ == "__main__":
  import numpy as np
  a = np.array([[1, 2, 3], [4, 5, 6]], order="F")
  print(a[0][0]) # output ==> 1
```  

Would be translated to this:  

```c
int main()
{
    t_ndarray a = {.shape = NULL};
    a = array_create(2, (int64_t[]){INT64_C(2), INT64_C(3)}, nd_int64, false, order_f); // Allocating the required ndarray
    t_ndarray temp_array = {.shape = NULL};
    temp_array = array_create(2, (int64_t[]){INT64_C(2), INT64_C(3)}, nd_int64, false, order_c); // Allocating an order_c temp_array
    int64_t array_dummy[] = {INT64_C(1), INT64_C(2), INT64_C(3), INT64_C(4), INT64_C(5), INT64_C(6)}; // array_dummy with our flattened data
    memcpy(temp_array.nd_int64, array_dummy, 6 * temp_array.type_size); // Copying our array_dummy to our temp ndarray
    array_copy_data(&a, temp_array, 0); // Copying into a column-major memory layout
    free_array(temp_array); // Freeing the temp_array right after we were done with it
    printf("%ld\n", GET_ELEMENT(a, nd_int64, (int64_t)0, (int64_t)0)); // output ==> 1
    free_array(a);
    return 0;
}
```

If the data is composed of at least one variable array, the process would still be somewhat the same as an `order_c ndarray` creation:
   The `order_f ndarray` is not populated from the get go, instead, we create an `order_c temp_array` (following `order_c ndarray` creation steps) containing all the data, then we do a 'copy into a column-major memory layout' operation to our `order_f ndarray`.  

Example:  

```python
if __name__ == "__main__":
  import numpy as np
  a = np.array([1, 2, 3])
  b = np.array([4, 5, 6])
  f = np.array([a, [7, 8, 9], b], order="F")
``` 

Would be translated to (focus on `f` `ndarray` creation):

```c
int main()
{
    t_ndarray a = {.shape = NULL};
    t_ndarray b = {.shape = NULL};
    t_ndarray c = {.shape = NULL};
    a = array_create(1, (int64_t[]){3}, nd_int64, false, order_c);
    int64_t array_dummy[] = {INT64_C(1), INT64_C(2), INT64_C(3)};
    memcpy(a.nd_int64, array_dummy, 3 * a.type_size);
    b = array_create(1, (int64_t[]){INT64_C(3)}, nd_int64, false, order_c);
    int64_t array_dummy_0001[] = {INT64_C(4), INT64_C(5), INT64_C(6)};
    memcpy(b.nd_int64, array_dummy_0001, 3 * b.type_size);
    
    // 'f' ndarray creation
    
    f = array_create(2, (int64_t[]){INT64_C(3), INT64_C(3)}, nd_int64, false, order_f); // Allocating the required ndarray (order_f)
    t_ndarray temp_array = {.shape = NULL};
    temp_array = array_create(2, (int64_t[]){INT64_C(3), INT64_C(3)}, nd_int64, false, order_c); // Allocating a temp_array (order_c)
    uint32_t offset = 0;
    array_copy_data(&temp_array, a, offset); // Copying the first element to temp_array
    offset += a.length;
    int64_t array_dummy_0002[] = {INT64_C(7), INT64_C(8), INT64_C(9)};
    memcpy(temp_array.nd_int64 + offset, array_dummy_0002, 3 * temp_array.type_size); // Copying the second element to temp_array
    offset += 3;
    array_copy_data(&temp_array, b, offset); // Copying the third element to temp_array
    array_copy_data(&f, temp_array, 0); // Copying our temp_array into a column-major memory layout (order_f)
    free_array(temp_array); // freeing the temp_array
    free_array(a);
    free_array(b);
    free_array(c);
    return 0;
}
```
