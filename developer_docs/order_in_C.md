# Multidimensional ndarrays memory layout (order)

## Order  

`order` is the parameter given the `numpy.array` in order to choose how the array is stored in memory, both  `Pyccel` supported orders are stored contiguously in memory, they differ in the order - the order of the values -.
`order='F'` would tell `numpy` to store the array column by column (column-major), example:
```python
import numpy as np

if __name__ == "__main__":
  a = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]], order='F')
  print(a.ravel('K'))
```
`array.ravel('k')` shows us how the array is actually stored in memroy, this python program will output `[1 4 7 2 5 8 3 6 9]`, notice that the columns are stored one after the other.  

`order='C'` on the other hand would tell `numpy` to store the array row by row, example:
```python
import numpy as np

if __name__ == "__main__":
  a = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]], order='C') # order='C' is the default in numpy.array
  print(a.ravel('K'))
 ```
This python program will output `[1 2 3 4 5 6 7 8 9]`, notice that the rows are stored one after the other.

### printing and indexing in `numpy`

`order` in `numpy` doesn't not affect the indexing or the printing, unlike `transposing`, the `shape` of the array remains the same, only the `strides` change, example:
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
`arr.strides` is how the printing and indexing occures, the strides of an array tell us how many bytes we have to skip in memory to move to the next position along a certain axis (dimension). For example for `memory_layout_of_a = [1 4 7 2 5 8]` and `strides_of_a = (8, 24)`, we have to skip 8 bytes (1 value for `int64`) to move to the next row, but 24 bytes (3 values for `int64`) to get to the same position in the next column of `a`.  
`a[2][1]` would give us `'8'`, using the `strides`: `2 * 8 + 1 * 24 = 40`, which means that in the flattened array, we would have to skip `40` bytes to get the value of `a[2][1]`, each element is 8 bytes, so we would have to skip `40 / 8 = 5` elements, from `1` to `5` to get to `'8'`


### Ordering in C
### Indexing in C
// Should it work the same as numpy

### Ordering in Fortran
// TODO

## ndarrays
`ndarrays` are stored as contiguous data in memory to increase efficiency
We use `copy_array_data(t_ndarray *dest, t_ndarray *src, t_uint32 offset, enum order)` to achieve this whether the order is `order_f` or `order_c`.

Taking as an example: `ab = array([a, b])` where `a = array([1, 2, 3])` and `b = array([4, 5, 6])`  

`ab` would be stored like this in `order_c`:


![C order memory layout](media/c_order_memory_layout.png)  

and like this in `order_f`:  

![F order memory layout](media/f_order_memory_layout.png)  
